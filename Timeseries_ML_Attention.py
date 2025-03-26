import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Layer, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.impute import SimpleImputer

#------------------------------#
# Data Loading / Preprocessing #
#------------------------------#
def load_and_prepare_data(phase_csv, outcome_csv):

    phase_wide = pd.read_csv(phase_csv)
    outcome_data = pd.read_csv(outcome_csv, usecols=['subject_id', 'dep_anx_composite.Final'])
    data_merged = pd.merge(phase_wide, outcome_data, on='subject_id')

    #Drop non-feature columns
    x_data = data_merged.drop(columns=['subject_id', 'dep_anx_composite.Final'])

    # Select drinking data (PreBaseline or Treatment)
    drinking_cols = [col for col in x_data.columns
                     if col.startswith("PreBaseline_") or col.startswith("Treatment_")]

    # Arrange by day
    drinking_cols_sorted = sorted(drinking_cols, key=lambda x: int(x.split('_')[-1]))
    x_data_sorted = x_data[drinking_cols_sorted]
    x_array = x_data_sorted.to_numpy()

    # Get outcome to predict
    y_data = data_merged['dep_anx_composite.Final']

    # Reshape feature data [samples, timesteps(days), features]
    num_samples, num_timesteps = x_data_sorted.shape
    x_array = x_array.reshape(num_samples, num_timesteps, 1)
    print("x_array shape:", x_array.shape)

    # Mean impute missing data
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    x_array_reshaped = x_array.reshape(-1, 1)
    x_imputed = imputer.fit_transform(x_array_reshaped)
    x_array_imputed = x_imputed.reshape(num_samples, num_timesteps, 1)

    return x_array_imputed, y_data, num_timesteps


# ------------------------------#
#     SHAP Attention Layer      #
# ------------------------------#
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
    lstm_out_1 = LSTM(30, return_sequences=True, name='lstm_layer_1')(input_layer)
    lstm_out_2 = LSTM(15, return_sequences=True, name='lstm_layer_2')(lstm_out_1)

    # Regularization and normalization
    dropout_layer = Dropout(0.1, name='dropout_layer')(lstm_out_2)
    batch_norm = BatchNormalization(name='batch_norm')(dropout_layer)

    # Attention Function
    attention_out = SimpleAttention(name='simple_attention')(batch_norm)

    # Dense/Output Layer
    dense_out_1 = Dense(30, activation='relu', name='dense_layer_1')(attention_out)
    dense_out_2 = Dense(20, activation='relu', name='dense_layer_2')(dense_out_1)
    output_layer = Dense(1, name='output_layer')(dense_out_2)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse')

    model.summary()
    return model


#------------------------------#
#    SHAP Analysis Function    #
#------------------------------#
def perform_shap_analysis(model, x_data, num_timesteps, features=1, background_samples=10):

    # Flatten data to vector
    x_flat = x_data.reshape(x_data.shape[0], -1)
    background_flat = x_flat[:background_samples]

    def model_predict_wrapper(x):
        # Reshape the flat input back [samples, timesteps, features]
        x_reshaped = x.reshape(-1, num_timesteps, features)
        return model.predict(x_reshaped)

    explainer = shap.KernelExplainer(model_predict_wrapper, background_flat)
    shap_values = explainer.shap_values(background_flat)

    # SHAP summary plot
    shap.summary_plot(shap_values, background_flat)
    plt.savefig("shap_summary_plot.png")
    plt.show()

    # If shap_values is a 3D array and features == 1, squeeze the last dimension
    shap_vals_squeezed = np.squeeze(shap_values)  # Expected shape: (samples, timesteps)
    mean_abs_shap = np.mean(np.abs(shap_vals_squeezed), axis=0)
    most_influential_index = np.argmax(mean_abs_shap)
    print(f"Most influential time step index: {most_influential_index}, "
          f"with mean absolute SHAP value: {mean_abs_shap[most_influential_index]}")

    return shap_values


def perform_shap_analysis_robust(model, x_data, num_timesteps, features=1,
                                 background_samples=50, nsamples=200, iterations=3):
    # Flatten x_data for  explainer (each sample is a flat vector)
    x_flat = x_data.reshape(x_data.shape[0], -1)
    shap_values_list = []

    for i in range(iterations):
        # Randomly select a larger background sample
        idx = np.random.choice(x_flat.shape[0], background_samples, replace=False)
        background_flat = x_flat[idx]

        # Define a wrapper to reshape flat inputs back to the original shape
        def model_predict_wrapper(x):
            x_reshaped = x.reshape(-1, num_timesteps, features)
            return model.predict(x_reshaped)

        # KernelExplainer with background sample
        explainer = shap.KernelExplainer(model_predict_wrapper, background_flat)
        shap_vals = explainer.shap_values(x_flat, nsamples=nsamples)
        shap_values_list.append(shap_vals)

    # Average the SHAP values from all iterations
    avg_shap_values = np.mean(shap_values_list, axis=0)

    # Plot and save the SHAP summary
    shap.summary_plot(avg_shap_values, x_flat)
    plt.savefig("shap_summary_plot_robust.png")
    plt.show()

    # If the output is 3D (samples x timesteps x features), squeeze the features dimension
    shap_vals_squeezed = np.squeeze(avg_shap_values)
    mean_abs_shap = np.mean(np.abs(shap_vals_squeezed), axis=0)
    most_influential_index = np.argmax(mean_abs_shap)
    print(f"Most influential time step index: {most_influential_index}, "
          f"with mean absolute SHAP value: {mean_abs_shap[most_influential_index]}")

    return avg_shap_values


#------------------------------#
#     Run Model and SHAP       #
#------------------------------#
if __name__ == "__main__":
    #Paths to data
    phase_csv = '/Users/bobkohler/Desktop/Manuscripts/NCIG Machine Learning/Prebaseline Effect Manuscript Materials/ncig_tlfb_allphase.csv'
    outcome_csv = '/Users/bobkohler/Desktop/Manuscripts/NCIG Machine Learning/Bob Manuscript Materials/NCIG1_Data/ncig_allvars_shared_select_tx_merged_redo3.csv'

    #Load and Preprocess
    x_array_imputed, y_data, num_timesteps = load_and_prepare_data(phase_csv, outcome_csv)

    #Model Build
    model = build_lstm_model(num_timesteps, features=1)

    #Model Train
    history = model.fit(x_array_imputed, y_data, epochs=100, batch_size=32, validation_split=0.2)

    #Model Loss
    loss = model.evaluate(x_array_imputed, y_data)
    print("Evaluation Loss:", loss)

    #Show predictions for first 10 samples
    predictions = model.predict(x_array_imputed[:10])
    comparison_df = pd.DataFrame({
        'Actual': y_data.iloc[:10].values,
        'Prediction': predictions.flatten()
    })
    print(comparison_df)

    # SHAP
    perform_shap_analysis_robust(model, x_array_imputed, num_timesteps,
                              features=1, background_samples=50, nsamples=20, iterations=3)

    plot_shap_time_importances(avg_shap_values, num_timesteps)