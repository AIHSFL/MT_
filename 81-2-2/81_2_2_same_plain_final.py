# -*- coding: utf-8 -*-
'''
Script:81-2-2
Daten: CTR-Mode, same plaintext
Model: Input[c,c_features] | Output[p,k]
Ziel: Vorhersage von p und k
Validierung: Separater Datensatz
Ergebnisse werden exportiert
'''

#-1---Imports-------------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import joblib
import os
from google.colab import files
from IPython.display import SVG
from tensorflow.keras.utils import model_to_dot
from tensorflow.keras.utils import plot_model
import platform
import psutil
import shutil
import sys
from datetime import datetime

#-2---Settings------------------------------------------------------------------
path_training_data = "/content/87F_CTR_p-nr-lett-8-By_k-8-By_1000000_same-plaintext.csv"
path_test_data="/content/87F_CTR_p-nr-lett-8-By_k-8-By_1000_same-plaintext.csv"

prefix = "81-2-2"

model_name = f"{prefix}-MLP.h5"
model_name_ = f"{prefix}-MLP.keras"

results_traindata_filename=f"{prefix}-prediction_on_train-ds.csv"
results_validdata_filename=f"{prefix}-prediction_on_validat-ds.csv"
model_sruct=f"{prefix}-model_structure.png"

BIT_LENGTH = 64

#-3---Hilsfunktionen------------------------------------------------------------

#---Datenvorbereitung
def hex_to_bin_vector(hex_str, length=BIT_LENGTH):
    bin_str = bin(int(hex_str, 16))[2:].zfill(length)
    return np.array([int(b) for b in bin_str])

def extract_features_from_c_advanced(hex_str, bit_length=BIT_LENGTH):
    bin_vec = hex_to_bin_vector(hex_str, bit_length)
    features = []

    #-1--Roh-Bits
    features.extend(bin_vec)

    #-2--Popcount & Parität
    popcount = np.sum(bin_vec)
    features.append(popcount / bit_length)
    features.append(popcount % 2)

    #-3--Quartile (lokale Popcounts)
    q = bit_length // 4
    for i in range(4):
        features.append(np.sum(bin_vec[i*q:(i+1)*q]) / q)

    #-4--Bit-Flips (0<->1)
    bit_flips = sum(bin_vec[i] != bin_vec[i+1] for i in range(bit_length - 1))
    features.append(bit_flips / (bit_length - 1))

    #-5--XOR benachbarter Bits
    for i in range(bit_length - 1):
        features.append(bin_vec[i] ^ bin_vec[i+1])

    #-6--AND benachbarter Bits
    for i in range(bit_length - 1):
        features.append(bin_vec[i] & bin_vec[i+1])

    #-7--S-Box Pattern: 4-Bit Nibbles
    for i in range(0, bit_length, 4):
        nibble = bin_vec[i:i+4]
        if len(nibble) == 4:
            index = int("".join(str(b) for b in nibble), 2)
            features.append(index / 15)

    #-8--Shifted Inputs (zirkular)
    features.extend(np.roll(bin_vec, -1))  # Linksrotation
    features.extend(np.roll(bin_vec, 1))   # Rechtsrotation

    return np.array(features)

#64 Bits erwartet:Hex-String = 64/4 = 16 Zeichen
def is_valid_hex(hex_str, bit_length=BIT_LENGTH):
    try:
        if len(hex_str) != bit_length // 4:
            return False
        int(hex_str, 16)
        return True
    except:
        return False

#---Auswertung für Vorhersage

#---Funktionen für binär-hex
def binvec_to_hex(binvec):
    bits_str = ''.join(str(int(b)) for b in binvec)
    return hex(int(bits_str, 2))[2:].zfill(len(binvec) // 4)

def bit_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100

#---Ähnlichkeit in Nibble-Blöcken (4-Bit-Gruppen)
def compare_nibbles(a, b):
    return sum(int("".join(map(str, a[i:i+4])), 2) ==
               int("".join(map(str, b[i:i+4])), 2)
               for i in range(0, len(a), 4))

#---Informationen
def print_model_keys():
  print(history.history.keys())

#---Speichern
def save_and_download_graphic(filename):
  plot_filename = f"{filename}.png"
  plt.savefig(plot_filename)
  plt.show()

  #-Herunterladen
  files.download(plot_filename)

def get_and_save_system_info():
    #-Speichern

    #-GPU Info
    gpu_info = !nvidia-smi
    gpu_info = '\n'.join(gpu_info)

    output = f"System Information Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    output += "="*60 + "\n\n"

    #-GPU Info
    output += "GPU Info:\n"
    if gpu_info.find('failed') >= 0:
        output += 'Not connected to a GPU\n'
    else:
        output += gpu_info + '\n'
    output += "\n"

    #-RAM Info
    ram = psutil.virtual_memory()
    ram_gb = ram.total / 1e9
    output += "RAM Info:\n"
    output += f'Total RAM: {ram_gb:.1f} GB\n'
    output += 'High-RAM runtime detected\n' if ram_gb >= 20 else 'Not using a high-RAM runtime\n'
    output += "\n"

    #-CPU Info
    cpu_freq = psutil.cpu_freq()
    output += "CPU Info:\n"
    output += f"Processor: {platform.processor() or 'Unknown'}\n"
    output += f"Logical Cores: {psutil.cpu_count(logical=True)}\n"
    output += f"Physical Cores: {psutil.cpu_count(logical=False)}\n"
    output += f"Max Frequency: {cpu_freq.max:.2f} MHz\n"
    output += f"Current Frequency: {cpu_freq.current:.2f} MHz\n\n"

    #-Disk Info
    disk_usage = shutil.disk_usage('/')
    total_disk = disk_usage.total / 1e9
    free_disk = disk_usage.free / 1e9
    output += "Disk Info:\n"
    output += f"Total Disk Size: {total_disk:.1f} GB\n"
    output += f"Free Space: {free_disk:.1f} GB\n\n"

    #-OS + Python Info
    output += "System Environment:\n"
    output += f"OS: {platform.system()} {platform.release()} ({platform.version()})\n"
    output += f"Python Version: {sys.version}\n"

    #-Speichern
    with open(f"{prefix}-system_info.txt", 'w') as f:
        f.write(output)

    #-Download

    files.download(f"{prefix}-system_info.txt")

def save_history(history):
  #-Zugriff auf das Dictionary
  history_file_name=f"{prefix}-Training_summary.txt"

  #hist = history.history
  epochs = len(history[list(history.keys())[0]])

  #-Textinhalt aufbauen
  summary = f"Training Summary\n{'='*40}\n"
  summary += f"Total Epochs: {epochs}\n\n"

  for key in history:
      values = history[key]
      summary += f"  {key}\n"
      summary += f"- Final Value: {values[-1]:.4f}\n"
      summary += f"- Max Value: {max(values):.4f}\n"
      summary += f"- Min Value: {min(values):.4f}\n"
      summary += f"- Average: {np.mean(values):.4f}\n\n"

  #-Speichern
  with open(history_file_name, 'w') as f:
      f.write(summary)

  #-Download
  files.download(history_file_name)

#-4---Daten einlesen------------------------------------------------------------
df = pd.read_csv(path_training_data)

#---Nur gültige Zeilen behalten-------------------------------------------------
valid_rows = df.apply(lambda row:
                      is_valid_hex(row['c']) and
                      is_valid_hex(row['p']) and
                      is_valid_hex(row['k']), axis=1)

df = df[valid_rows].reset_index(drop=True)
print(f"Nach Filterung bleiben {len(df)} Zeilen übrig")

#---Feature-Extraktion----------------------------------------------------------
X_c = np.vstack(df['c'].apply(extract_features_from_c_advanced).values)
X_p = np.vstack(df['p'].apply(hex_to_bin_vector).values)
X_k = np.vstack(df['k'].apply(hex_to_bin_vector).values)

X = np.hstack([X_c])
y_p = X_p.copy()
y_k = X_k.copy()

print("Feature-Extraktion erfolgreich abgeschlossen.")

#-5---Modellaufbau--------------------------------------------------------------
input_layer = Input(shape=(X.shape[1],),
                    name='input_all')

x = Dense(512, activation='relu')(input_layer)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)

p_out = Dense(BIT_LENGTH, activation='sigmoid', name='p_out')(x)
k_out = Dense(BIT_LENGTH, activation='sigmoid', name='k_out')(x)

model = Model(inputs=input_layer,
              outputs=[p_out, k_out])

#Metrik pro Output explizit angeben:
#Getrennte Berechnung von accuracy für p und k

model.compile(
    optimizer=Adam(1e-3),
    loss={
        'p_out': 'binary_crossentropy',
        'k_out': 'binary_crossentropy'
    },
    metrics={
    'p_out': [BinaryAccuracy(), Precision(), Recall()],
    'k_out': [BinaryAccuracy(), Precision(), Recall()]
    }
)
callbacks = [
    EarlyStopping(monitor='val_loss',
                  patience=3,
                  restore_best_weights=True,
                  mode='min'),
    ReduceLROnPlateau(monitor='val_loss',
                      factor=0.5,
                      patience=3,
                      mode='min')
]

model.summary()

#-6---Training------------------------------------------------------------------
X_train, X_val, y_p_train, y_p_val, y_k_train, y_k_val = train_test_split(
    X, y_p, y_k, test_size=0.2, random_state=42)

history = model.fit(
    X_train, [y_p_train, y_k_train],
    validation_data=(X_val, [y_p_val, y_k_val]),
    epochs=500,
    batch_size=64,
    callbacks=callbacks
)

#-7---Vorhersage----------------------------------------------------------------

#-1--C-Features Anzahl aus dem Val extrahieren---
X_val_c = X_val[:, :X_c.shape[1]]

#-2--Eingabe für Vorhersage zusammenbauen---------------------------------------
X_val_pred_input = np.hstack([X_val_c])

#-4--Vorhersagen generieren-----------------------------------------------------
y_p_pred, y_k_pred = model.predict(X_val_pred_input)

#-5--Binarisieren---------------------------------------------------------------

y_p_bin = (y_p_pred > 0.5).astype(int)
y_k_bin = (y_k_pred > 0.5).astype(int)

#-6--Ausgabe je Sample----------------------------------------------------------
for i in range(len(X_val)):
    # p
    true_p_hex = binvec_to_hex(y_p_val[i])
    pred_p_hex = binvec_to_hex(y_p_bin[i])
    acc_p = bit_accuracy(y_p_val[i], y_p_bin[i])

    # k
    true_k_hex = binvec_to_hex(y_k_val[i])
    pred_k_hex = binvec_to_hex(y_k_bin[i])
    acc_k = bit_accuracy(y_k_val[i], y_k_bin[i])

#-7--Als Dataframe--------------------------------------------------------------
results = []
for i in range(len(X_val)):
    results.append({
        'c': df['c'].iloc[i],
        'p_true': binvec_to_hex(y_p_val[i]),
        'p_true_bin': y_p_val[i],
        'p_pred': binvec_to_hex(y_p_bin[i]),
        'p_pred_bin': y_p_bin[i],
        'p_acc': bit_accuracy(y_p_val[i], y_p_bin[i]),
        'k_true': binvec_to_hex(y_k_val[i]),
        'k_true_bin': y_k_val[i],
        'k_pred': binvec_to_hex(y_k_bin[i]),
        'k_pred_bin': y_k_bin[i],
        'k_acc': bit_accuracy(y_k_val[i], y_k_bin[i]),
    })

results_df = pd.DataFrame(results)
results_df.head()

results_df.to_csv(results_traindata_filename, index=False)
print(" Vorgersage an Trainingsdatensatz abgeschlossen")

#-8---Vorhersage auf Test-CSV---------------------------------------------------
test_df = pd.read_csv(path_test_data)
X_test_c = np.vstack(test_df['c'].apply(extract_features_from_c_advanced).values)
X_test_p = np.vstack(test_df['p'].apply(hex_to_bin_vector).values)
X_test_k = np.vstack(test_df['k'].apply(hex_to_bin_vector).values)

X_test = np.hstack([X_test_c])

y_pred_p, y_pred_k = model.predict(X_test)

#-Binarisieren
y_pred_p_bin = (y_pred_p > 0.5).astype(int)
y_pred_k_bin = (y_pred_k > 0.5).astype(int)

# ----------------------------
results_test = []
#-Vergleich & Ausgabe
for i in range(len(test_df)):
    p_acc = np.mean(y_pred_p_bin[i] == X_test_p[i]) * 100
    k_acc = np.mean(y_pred_k_bin[i] == X_test_k[i]) * 100
    print(p_acc, k_acc)

    results_test.append({
          'c': test_df['c'].iloc[i],
          'p_true': test_df['p'].iloc[i],
          'p_true_bin': X_test_p[i],
          'p_pred': binvec_to_hex(y_pred_p_bin[i]),
          'p_pred_bin': y_pred_p_bin[i],
          'p_acc': p_acc,
          'k_true': test_df['k'].iloc[i],
          'k_true_bin': X_test_k [i],
          'k_pred': binvec_to_hex(y_pred_k_bin[i]),
          'k_pred_bin': y_pred_k_bin[i],
          'k_acc': k_acc,
    })

results_test_df = pd.DataFrame(results_test)
results_test_df.head()

results_test_df.to_csv(results_validdata_filename, index=False)
print(" Erstellt")

'''
print(f"Sample {i+1}:")
print(f"  p: true={true_p}, pred={pred_p}, accuracy={p_acc:.1f}%")
print(f"  k: true={true_k}, pred={pred_k}, accuracy={k_acc:.1f}%\n")
'''

#---Download--------------------------------------------------------------------

#-Speichern
model.save(model_name)
model.save(model_name_)

#-Download des models
files.download(model_name)
files.download(model_name_)

#-Download results train-dataset
results_df.to_csv(f"{results_traindata_filename}", index=False)
print(" Ergebnisse als" f"{results_traindata_filename}" " gespeichert.")
files.download(f"{results_traindata_filename}")

#-Download results validation-dataset
results_test_df.to_csv(f"{results_validdata_filename}", index=False)
print(" Ergebnisse als" f"{results_validdata_filename}" " gespeichert.")
files.download(f"{results_validdata_filename}")

#---Trainingsverlauf plotten---
#-Plot erstellen
plt.figure(figsize=(10, 6))
p_k_all_acc_name=f"{prefix}-P, K, Gesamt- Loss Verlauf"

plt.plot(history.history['p_out_binary_accuracy'], label='p_out_bin-acc')
plt.plot(history.history['p_out_loss'], label='p_out_loss')
plt.plot(history.history['k_out_binary_accuracy'], label='k_out_bin acc')
plt.plot(history.history['k_out_loss'], label='k_out_loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy / Loss')
plt.legend()
plt.grid(True)
plt.title('P_K-Accuracy-Loss')

save_and_download_graphic(p_k_all_acc_name)

#---Loss Verlauf---
loss_name=f"{prefix}- P-K-Gesamt-Loss_Verlauf"
plt.figure(figsize=(10, 6))

plt.plot(history.history['loss'], label='Trainings-Loss')
plt.plot(history.history['val_loss'], label='Validierungs-Loss')
plt.plot(history.history['k_out_loss'], label='k_out_loss')
plt.plot(history.history['p_out_loss'], label='p_out_loss')

plt.plot(history.history['val_k_out_loss'], label='val_k_out_loss')
plt.plot(history.history['val_p_out_loss'], label='val_p_out_loss')

plt.legend()
plt.title(loss_name)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

save_and_download_graphic(loss_name)

#---Histogramm der Bitgenauigkeit---
bit_acc_name=f"{prefix}-Bitgenauigkeit"
plt.figure(figsize=(10, 6))

results_df['p_acc'].hist(bins=20, alpha=0.7, label='p')
results_df['k_acc'].hist(bins=20, alpha=0.7, label='k')
plt.xlabel("Bitgenaue Übereinstimmung (%)")
plt.ylabel("Anzahl Samples")
plt.title("Verteilung der Bit-Genauigkeit")
plt.legend()
plt.grid(True)

save_and_download_graphic(bit_acc_name)

#---Modelvisualisierung---
#-Netz mit allen Schichten, Namen und Output-Formaten
SVG(model_to_dot(model, show_shapes=True, show_layer_names=True).create(prog='dot', format='svg'))

#-Modell speichern
plot_model(model, to_file=model_sruct, show_shapes=True, show_layer_names=True, dpi=100)

files.download(model_sruct)

#---Weitere Parameter plotten---
#-experimentell
precis_recall_name=f"{prefix}-p-k_precis_recall"
plt.figure(figsize=(10, 6))

plt.plot(history.history['k_out_precision_1'], label='k_out_precision_1')
plt.plot(history.history['p_out_precision'], label='p_out_precision')

plt.plot(history.history['k_out_recall_1'], label='k_out_recall_1')
plt.plot(history.history['p_out_recall'], label='p_out_recall')

plt.plot(history.history['learning_rate'], label='learning_rate')

plt.plot(history.history['val_k_out_binary_accuracy'], label='val_k_out_binary_accuracy')

plt.plot(history.history['val_p_out_binary_accuracy'], label='val_p_out_binary_accuracy')


plt.plot(history.history['p_out_precision'], label='p_out_precision')


plt.xlabel('Epoch')
plt.ylabel('Precision, Recall')
plt.legend()
plt.grid(True)
plt.title(precis_recall_name)

save_and_download_graphic(precis_recall_name)

get_and_save_system_info()

save_history(history.history)

print_model_keys()