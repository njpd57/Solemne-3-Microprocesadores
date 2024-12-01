print("[+]Cargando librerias")
import tensorflow as tf
import numpy as np
import joblib

print("[+]Definiendo Variables")
MODEL_PATH ="model.keras"
SCALER_PATH ="target-scaler.gz"
DATA_VALUES = "X-test.npy"
OUTPUT_PATH = "output.npy"
OUTPUT_CSV_PATH = "output.csv"

print("[+]Cargando Valores de Prueba")
X_test = np.load(DATA_VALUES)
print("[+]Cargando Escalador")
target_scaler = joblib.load(SCALER_PATH)
print("[+]Cargando Modelo")
model = tf.keras.models.load_model(MODEL_PATH)

print("[+]Comienza Predicción")
y_pred = model.predict(X_test)

print("[+]Denormalizando Valores")
y_pred = target_scaler.inverse_transform(y_pred[0])

print("[+]Guardando Resultados")
np.save(OUTPUT_PATH,y_pred)
np.savetxt(OUTPUT_CSV_PATH,y_pred,delimiter=",")

print("[+]Fin del Proceso")





