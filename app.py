import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import numpy as np

data = pd.read_csv("/content/drive/MyDrive/Bias_correction_ucl.csv")
if data.isnull().values.any():
    data = data.dropna()
feature_columns = [
    'Present_Tmax', 'Present_Tmin', 'LDAPS_RHmin', 'LDAPS_RHmax',
    'LDAPS_Tmax_lapse', 'LDAPS_Tmin_lapse', 'LDAPS_WS', 'LDAPS_LH',
    'lat', 'lon', 'DEM', 'Slope', 'Solar radiation'
]
target_columns = ['Next_Tmax', 'Next_Tmin']
X = data[feature_columns]
y = data[target_columns]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(2)])

model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train_scaled, y_train, epochs=50, validation_split=0.2, batch_size=32, verbose=1)

y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)

st.title('Temperature Prediction for Seoul, South Korea')
st.write('Enter the following details to predict next day\'s maximum and minimum temperatures.')
st.write(f'Mean Squared Error on the test set: {mse:.2f}')

user_input = {}
for feature in feature_columns:
    user_input[feature] = st.number_input(f'Enter {feature}:', value=float(data[feature].mean()))

user_df = pd.DataFrame(user_input, index=[0])
user_scaled = scaler.transform(user_df)

if st.button('Predict'):
    prediction = model.predict(user_scaled)
    st.write(f'Predicted Next Day Maximum Temperature: {prediction[0][0]:.2f}°C')
    st.write(f'Predicted Next Day Minimum Temperature: {prediction[0][1]:.2f}°C')
    st.write(f'User input after scaling: {user_scaled}')
    st.write(f'Model summary:')
    model.summary(print_fn=lambda x: st.text(x))
    st.write(f'History of loss values: {history.history["loss"]}')
