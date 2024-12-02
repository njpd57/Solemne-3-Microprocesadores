print("[+] Cargar Libresias")
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler

#Variables de Ruta
OUTPUT_FILES    = "out"
DATAFRAME_PATH  = "data.csv"
IMAGE_PATH      = f"{OUTPUT_FILES}/IMAGES"
SCALER_OUTPUT   = f"{OUTPUT_FILES}/SCALER/target-scaler.gz"
MACHINE_MODEL_PATH  = f"{OUTPUT_FILES}/MACHINE_MODEL/model.gz"

#Variables de Entrenamiento
TRAIN_SIZE      = 0.8   # 80% Entrenamiento - 20% Pruebas

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
scaler = MinMaxScaler(feature_range=(-1,1))
df_scaled = scaler.fit_transform(df_clean.to_numpy())
df_scaled = pd.DataFrame(df_scaled, columns=list(df_clean.columns))

objetivo = df_clean["temperatura"].to_numpy()
objetivo = objetivo.reshape(-1,1)

target_scaler = MinMaxScaler(feature_range=(-1, 1))
df_scaled['temperatura'] = target_scaler.fit_transform(objetivo)
df_scaled = df_scaled.astype(float)

#Exportar Scaler para futura predicción
joblib.dump(target_scaler,SCALER_OUTPUT)

print("[+] Preparación de Datos")
#Separación de datos
X = df_scaled.drop(columns=["temperatura"])
y = df_scaled["temperatura"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=False)

print("[+] Compilación de Modelo")
#Implementación de Modelo
model = DecisionTreeRegressor()

print("[+] Comienza Entrenamiento")
#Inicio de Entrenamiento, early stop nos permitira entrenar hasta que el modelo deje de mejorar
model.fit(X_train,y_train)

print("[+] Fin de Entrenamiento")


print("[+] Análisis de Desempeño")
#Desempeño del Modelo
y_pred = model.predict(X_test)
print("MSE: ",mean_squared_error(y_test,y_pred))
print("MAE:",mean_absolute_error(y_test,y_pred))
print("R2 SCORE:",r2_score(y_test,y_pred))

y_pred=pd.Series(y_pred,index=y_test.index)

temperaturas = target_scaler.inverse_transform(y_test.values.reshape(-1,1))
prediccion = target_scaler.inverse_transform(y_pred.values.reshape(-1,1))
plt.figure(figsize=(10,6))
plt.plot(temperaturas,label="Valores Reales")
plt.plot(prediccion,label="Valores Predichos")
plt.ylabel("Prediccion")
plt.xlabel("Muestras")
plt.title("Desempeño del Modelo")
plt.legend(loc="upper right")
plt.savefig(f"{IMAGE_PATH}/prediccion.png")


print("[+] Exportando Modelo")
#Exportar Modelo
joblib.dump(model,MACHINE_MODEL_PATH,compress=3)

print("[+] Finalizando Proceso de manera éxitosa")