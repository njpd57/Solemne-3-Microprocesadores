print("[+] Cargar Librerias")
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

print("[+]Definiendo Variables")
MODEL_PATH      = "../../out/HIGH_LEVEL_MODEL/model.keras"
SCALER_PATH     = "../../out/SCALER/target-scaler.gz"

OUTPUT_PATH     = "output.npy"
OUTPUT_CSV_PATH = "output.csv"

N_PAST          = 3     # 3 Horas
N_FUTURE        = 5     # 5 Horas
N_OUTPUT        = 2     # Temperatura y Humedad
DATAFRAME_PATH  = "data.csv"

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



print("[+]Cargando DataFrame")
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

#PreProcesado
df_clean = df.drop(columns=["hora"])
df_clean = df_clean.reset_index(drop=True)

#Escalar Datos
scaler = MinMaxScaler(feature_range=(-1,1))
df_scaled = scaler.fit_transform(df_clean.to_numpy())
df_scaled = pd.DataFrame(df_scaled, columns=list(df_clean.columns))

target_scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled[['temperatura', 'humedad']] = target_scaler.fit_transform(df_clean[['temperatura', 'humedad']].to_numpy())
df_scaled = df_scaled.astype(float)

#Separación de datos
x = np.array([df_scaled[-N_PAST::]])

print("[+]Cargando Modelo")
model = tf.keras.models.load_model(MODEL_PATH)

print("[+]Comienza Predicción")
y_pred = model.predict(x)

print("[+]Denormalizando Valores")
y_pred = target_scaler.inverse_transform(y_pred[0])

start_date = df.index[-1].replace(second=0,microsecond=0)
hora = pd.date_range(start=start_date,periods=len(y_pred),freq='H')
hora= hora.hour

hora_df = pd.DataFrame(hora,columns=("Hora",))
df2 = pd.DataFrame(y_pred,columns=("Temperatura","Humedad"))
df2 = hora_df.join(df2)

print("[+]Guardando Resultados")
#np.savetxt(OUTPUT_CSV_PATH,y_pred,delimiter=",",fmt="%.1f",header="hora,temperatura,humedad",comments="")
df2.to_csv(OUTPUT_CSV_PATH,index=False,float_format="%.1f")

print("[+]Fin del Proceso")




