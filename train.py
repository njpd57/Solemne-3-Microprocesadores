print("[+] Cargar Librerias")
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Input, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf

#Variables de Ruta
OUTPUT_FILES    = "out"
DATAFRAME_PATH  = "data.csv"
IMAGE_PATH      = f"{OUTPUT_FILES}/IMAGES"
SCALER_OUTPUT   = f"{OUTPUT_FILES}/SCALER/target-scaler.gz"
H_MODEL_PATH    = f"{OUTPUT_FILES}/HIGH_LEVEL_MODEL/model.keras"
L_MODEL_PATH    = f"{OUTPUT_FILES}/LOW_LEVEL_MODEL/"
LITE_MODEL_PATH = f"{OUTPUT_FILES}/LITE_MODEL/model.tflite"

#Variables de Entrenamiento
TRAIN_SIZE      = 0.8   # 80% Entrenamiento - 20% Pruebas
N_PAST          = 3     # 3 Horas
N_FUTURE        = 5     # 5 Horas
N_OUTPUT        = 2     # Temperatura y Humedad
EPOCHS          = 2000  # Épocas


def split_series(series, n_past, n_future,columns):
    """
    Splits a time series into past and future windows.

    Args:
        series (numpy.ndarray): The time series to be split.
        n_past (int): The number of past observations in each window.
        n_future (int): The number of future observations in each window.

    Returns:
        tuple: A tuple of numpy arrays containing the past and future windows.
    """
    X, y = [], []  # initialize empty lists to store past and future windows
    for window_start in range(len(series)):
        past_end = window_start + n_past  # end index of past window
        future_end = past_end + n_future  # end index of future window
        if future_end > len(series):  # if future window extends beyond series, break loop
            break
        #print(window_start,past_end,future_end)

        past, future = series[window_start:past_end, :], series[past_end:future_end, :]  # slice past and future windows
        future = future[:][:,0:2] #Solamente temperatura y humedad
        X.append(past)
        y.append(future)
    return np.array(X), np.array(y)  # convert lists to numpy arrays and return as tuple




print("[+] Analisis Exploratorio de los datos")
# Leer dataframe
df = pd.read_csv(DATAFRAME_PATH)
# Establecer fecha de unix a datetime en ms
df["fecha"] = pd.to_datetime(df["fecha"],unit="ms")
# Establecer fecha inicial
start_date = df["fecha"][0].replace(second=0,microsecond=0)
# Reestablecer la fecha considerando solamente los minutos
df["fecha"] = pd.date_range(start=start_date,periods=len(df),freq='T')
# Almacenar hora
df['hora'] = df['fecha'].dt.hour
# Establecer indice como la fecha
df.set_index('fecha', inplace=True)
# Promediar por hora los datos
df_resampled = df.resample('60T').mean()
df = df_resampled
# Codificar la hora del día de manera cíclica
df['hora_sin'] = np.sin(2 * np.pi * df['hora'] / 24)  # 24 horas en un día
df['hora_cos'] = np.cos(2 * np.pi * df['hora'] / 24)

#Guardar Humedad Promediada
plt.figure(figsize=(10,6))
plt.plot(df["humedad"])
plt.xlabel("Fecha")
plt.ylabel("% de Humedad")
plt.title("Valores de Humedad")
plt.savefig(f"{IMAGE_PATH}/humedad.png")

#Guardar Temperatura Promediada
plt.figure(figsize=(10,6))
plt.plot(df["temperatura"])
plt.xlabel("Fecha")
plt.ylabel("Temperatura (C°)")
plt.title("Temperatura Obtenida por Sensor")
plt.savefig(f"{IMAGE_PATH}/temperatura.png")

#Ciclo dia y noche
plt.figure(figsize=(10,6))
plt.plot(df["hora_cos"])
plt.plot(df["hora_sin"])
plt.xlabel("Tiempo")
plt.ylabel("Seno / Coseno")
plt.title("Periodicidad del Tiempo")
plt.savefig(f"{IMAGE_PATH}/tiempo.png")


print("[+] Pre-Procesado de Datos")
#PreProcesado
df_clean = df.drop(columns=["hora"])
df_clean = df_clean.reset_index(drop=True)

#Escalar Datos
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1,1))
df_scaled = scaler.fit_transform(df_clean.to_numpy())
df_scaled = pd.DataFrame(df_scaled, columns=list(df_clean.columns))

target_scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled[['temperatura', 'humedad']] = target_scaler.fit_transform(df_clean[['temperatura', 'humedad']].to_numpy())
df_scaled = df_scaled.astype(float)

#Exportar Scaler para futura predicción
joblib.dump(target_scaler,SCALER_OUTPUT)

print("[+] Preparación de Datos")
#Separación de datos
split_index = round(len(df.index) * TRAIN_SIZE)
train, test = df_scaled[1:split_index], df_scaled[split_index:]

#Cantidad de variables
n_features = len(df_scaled.columns) 

#Conjunto de Entrenamiento
X_train, y_train = split_series(train.values,N_PAST, N_FUTURE,columns=("temperatura","humedad"))
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1],n_features))
y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], N_OUTPUT))

#Conjunto de Pruebas
X_test, y_test = split_series(test.values,N_PAST, N_FUTURE,columns=("temperatura","humedad"))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],n_features))
y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], N_OUTPUT))

print("[+] Compilación de Modelo")
#Implementación de Modelo
inputs = Input(shape=(N_PAST, n_features),batch_size=1)

#Long Short Term Memory
x1 = LSTM(20,activation='relu')(inputs)
x2 = Dense(N_FUTURE * N_OUTPUT)(x1)
outputs = Reshape ((N_FUTURE,N_OUTPUT))(x2) # 5 horas x variables
model = keras.Model(inputs=inputs, outputs=outputs)

#Compilar el modelo
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# Early stopping para monitorear la validación y detener el entrenamiento si no mejora
early_stop = EarlyStopping(monitor='mae', patience=5, restore_best_weights=True)


print("[+] Comienza Entrenamiento")
#Inicio de Entrenamiento, early stop nos permitira entrenar hasta que el modelo deje de mejorar
history = model.fit(X_train, y_train, epochs=EPOCHS,callbacks=early_stop)
print("[+] Fin de Entrenamiento")


print("[+] Análisis de Desempeño")
#Desempeño del Modelo
plt.figure(figsize=(10,6))
plt.plot(history.history["loss"],label="Pérdida")
plt.plot(history.history["mae"] ,label="Error Absoluto Medio")
plt.ylabel("Pérdida/MAE")
plt.xlabel("Épocas")
plt.title("Desempeño del Modelo")
plt.legend(loc="upper right")
plt.savefig(f"{IMAGE_PATH}/desempeño.png")

y_pred = model.predict(X_test)

#Transformar a unidimensional para una comparación más fácil a través de todos los pasos de tiempo.
y_true_temp = y_test[:, :, 0].flatten()  # Valores Reales Temperatura
y_pred_temp = y_pred[:, :, 0].flatten()  # Valores Predichos Temperatura

y_true_humidity = y_test[:, :, 1].flatten()  # Valores Reales  humedad
y_pred_humidity = y_pred[:, :, 1].flatten()  # Valores Predichos humedad

# RMSE para temperatura y humedad
rmse_temp = np.sqrt(mean_squared_error(y_true_temp, y_pred_temp))
rmse_humidity = np.sqrt(mean_squared_error(y_true_humidity, y_pred_humidity))

# MAE para temperatura y humedad
mae_temp = mean_absolute_error(y_true_temp, y_pred_temp)
mae_humidity = mean_absolute_error(y_true_humidity, y_pred_humidity)

# R² para temperatura y humedad
r2_temp = r2_score(y_true_temp, y_pred_temp)
r2_humidity = r2_score(y_true_humidity, y_pred_humidity)

print(f"Temperature Metrics:\n RMSE: {rmse_temp}, MAE: {mae_temp}, R²: {r2_temp}")
print(f"Humidity Metrics:\n RMSE: {rmse_humidity}, MAE: {mae_humidity}, R²: {r2_humidity}")


print("[+] Exportando Modelo API Alto Nivel")
#Exportar Modelo
model.save(H_MODEL_PATH)

print("[+] Exportando Modelo API Bajo Nivel")
tf.saved_model.save(model,L_MODEL_PATH)


print("[+] Exportando Modelo Tensorflow Lite")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Aplicar Optimización
converter.optimizations = [tf.lite.Optimize.DEFAULT]  
# Convertir modelo
tflite_model = converter.convert()

# Exportar Modelo
with open(f'{LITE_MODEL_PATH}', 'wb') as f:
    f.write(tflite_model)

print("[+] Finalizando Proceso de manera éxitosa")