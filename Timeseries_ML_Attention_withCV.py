import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Layer, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold


#------------------------------#
# Data Loading / Preprocessing #
#------------------------------#
def load_and_prepare_data(phase_csv, outcome_csv):

    phase_wide = pd.read_csv(phase_csv)
    outcome_data = pd.read_csv(outcome_csv, usecols=['subject_id', 'dep_anx_composite.Final', 'study'])
    data_merged = pd.merge(phase_wide, outcome_data, on='subject_id')

    # Drop non-feature columns
    x_data = data_merged.drop(columns=['subject_id', 'dep_anx_composite.Final'])

    # Get study site for cross validation
    groups = data_merged['study'].values

    # Select drinking data (PreBaseline or Treatment)
    drinking_cols = [col for col in x_data.columns
                     if col.startswith("PreBaseline_") or col.startswith("Treatment_")]
    # Arrange by day
    drinking_cols_sorted = sorted(drinking_cols, key=lambda x: int(x.split('_')[-1]))
    x_data_sorted = x_data[drinking_cols_sorted]
    x_array = x_data_sorted.to_numpy()

    # Get outcome to predict
    y_data = data_merged['dep_anx_composite.Final']

    # Reshape feature data [samples, timesteps, 1]
    num_samples, num_timesteps = x_array.shape
    x_array = x_array.reshape(num_samples, num_timesteps, 1)

    # Mean impute
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    x_reshaped = x_array.reshape(-1, 1)
    x_imputed = imputer.fit_transform(x_reshaped)
    x_array_imputed = x_imputed.reshape(num_samples, num_timesteps, 1)

    return x_array_imputed, y_data, groups, num_timesteps

#--------------------------------#
#        Attention Layer         #
#--------------------------------#
class SimpleAttention(Layer):
    def __init__(self, **kwargs):
        super(SimpleAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.attention_dense = Dense(1, activation='tanh')
        super(SimpleAttention, self).build(input_shape)

    def call(self, inputs):
        # Compute raw attention scores: (batch_size, timesteps, 1)
        scores = self.attention_dense(inputs)
        # Squeeze to shape: (batch_size, timesteps)
        scores = tf.squeeze(scores, axis=-1)
        # Compute normalized attention weights over timesteps
        weights = tf.nn.softmax(scores, axis=1)
        self.attention_weights = weights  # Save for later inspection if needed
        # Expand dims for weighted multiplication: (batch_size, timesteps, 1)
        weights_expanded = tf.expand_dims(weights, axis=-1)
        # Compute context vector as the weighted sum of inputs over time
        context = tf.reduce_sum(inputs * weights_expanded, axis=1)
        return context

#------------------------------#
#       Build LSTM Model       #
#------------------------------#
def build_lstm_model(num_timesteps, features=1):

    input_layer = Input(shape=(num_timesteps, features), name='input_layer')

    # LSTM layers w/ return sequence for time series
    lstm_out_1 = LSTM(20, return_sequences=True, name='lstm_layer_1')(input_layer)
    lstm_out_2 = LSTM(10, return_sequences=True, name='lstm_layer_2')(lstm_out_1)

    # Regularize/Normalize
    dropout_layer = Dropout(0.2, name='dropout_layer')(lstm_out_2)
    batch_norm = BatchNormalization(name='batch_norm')(dropout_layer)

    # Attention
    attention_out = SimpleAttention(name='simple_attention')(batch_norm)

    # Dense/Output Layer
    dense_out_1 = Dense(20, activation='relu', name='dense_layer_1')(attention_out)
    dense_out_2 = Dense(10, activation='relu', name='dense_layer_2')(dense_out_1)
    output_layer = Dense(1, name='output_layer')(dense_out_2)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse')

    model.summary()
    return model

#------------------------------#
#    SHAP Analysis Function    #
#------------------------------#

def perform_shap_analysis_robust(model, x_data, num_timesteps, features,
                                 background_samples, nsamples, iterations):

    # Flatten x_data for explainer
    x_flat = x_data.reshape(x_data.shape[0], -1)
    shap_values_list = []

    for i in range(iterations):
        # Randomly select background sample
        idx = np.random.choice(x_flat.shape[0], background_samples, replace=False)
        background_flat = x_flat[idx]

        # Define wrapper to reshape flat inputs to the original shape
        def model_predict_wrapper(x):
            x_reshaped = x.reshape(-1, num_timesteps, features)
            return model.predict(x_reshaped)

        # KernelExplainer with background sample
        explainer = shap.KernelExplainer(model_predict_wrapper, background_flat)
        shap_vals = explainer.shap_values(x_flat, nsamples=nsamples)
        shap_values_list.append(shap_vals)

    # Average SHAP values across all iterations
    avg_shap_values = np.mean(shap_values_list, axis=0)

    # Plot and save SHAP
    shap.summary_plot(avg_shap_values, x_flat)
    plt.savefig("shap_summary_plot_robust.png")
    plt.show()

    # If 3-D (samples x timesteps x features) then squeeze the features dimension
    shap_vals_squeezed = np.squeeze(avg_shap_values)
    mean_abs_shap = np.mean(np.abs(shap_vals_squeezed), axis=0)
    most_influential_index = np.argmax(mean_abs_shap)
    print(f"Most influential Day: {most_influential_index}, "
          f"with mean absolute SHAP value: {mean_abs_shap[most_influential_index]}")
    return avg_shap_values

#------------------------------------------------------#
#          Run Mode with Cross-Validation and SHAP     #
#------------------------------------------------------#

if __name__ == "__main__":
    phase_csv = '/Users/bobkohler/Desktop/Manuscripts/NCIG Machine Learning/Prebaseline Effect Manuscript Materials/ncig_tlfb_allphase.csv'
    outcome_csv = '/Users/bobkohler/Desktop/Manuscripts/NCIG Machine Learning/Bob Manuscript Materials/NCIG1_Data/ncig_allvars_shared_select_tx_merged_redo3.csv'

    x_array_imputed, y_data, groups, num_timesteps = load_and_prepare_data(phase_csv, outcome_csv)

    # Group-based cross-validation
    gkf = GroupKFold(n_splits=5)  # set splits

    fold = 1
    for train_idx, test_idx in gkf.split(x_array_imputed, y_data, groups=groups):
        print(f"=== Fold {fold} ===")
        fold += 1

        # Split data
        X_train, X_test = x_array_imputed[train_idx], x_array_imputed[test_idx]
        y_train, y_test = y_data.iloc[train_idx], y_data.iloc[test_idx]

        # Build a new model for each fold
        model = build_lstm_model(num_timesteps, features=1)

        # Train
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=0)

        # Evaluate on test data
        val_loss = model.evaluate(X_test, y_test, verbose=0)
        print(f"Fold {fold - 1} - Test Loss: {val_loss}")

    #----SHAP ANALYSIS----#
    # perform_shap_analysis_robust #
        # model = LSTM Output #
        # x_array_imputed = feature data across time that has been imputed. here, mean imputed #
        # num_timesteps = length of timeseries - here, number of prebaseline + baseline + treatment day #
        # features = number of features - here, the only feature is 'day'' (although across time) so features= 1 #
        # background_samples = number of samples from original dataset that are used simulate missing features when they are masked out for remodeling #
        # nsamples =  number of model re-evaluations when explaining which feature is driving prediction #
        # iterations = number of repeats for entire SHAP analysis #

    shap_values_robust =  perform_shap_analysis_robust(model,
                                 x_array_imputed,
                                 num_timesteps,
                                 features=1,
                                 background_samples=100,
                                 nsamples=200,
                                 iterations=1)
# Most influential time step index: 25, with mean absolute SHAP value: 0.004474957971854393
# background_samples = 100; nsamples = 200; iterations = 5)

  model.save("lstm_model.keras")