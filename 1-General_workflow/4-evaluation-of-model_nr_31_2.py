# -*- coding: utf-8 -*-

'''
Script: 31-2

Untersuchung der Modelvorhersagen auf belibig definierten 
(unbekanner oder bekannter Art)von Daten.

Ablauf: 
1. Definiere Datensettings
2. Generiere Daten
3. Lade das trainierte Modell
4. Verwende generierten Datensatz
5. Modell erstellt die Vorhersagen
6. Die Vorhersagen werden ausgewertet
7. Die Ergebnisse werden in einer Textdatei zusammengefasst
'''
#!pip install pycryptodome

#-1--Imports--------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import pandas as pd

from math import log2
from tensorflow.keras.models import load_model
from google.colab import files
import csv
from Crypto.Cipher import DES
from Crypto.Util.Padding import pad
import os
import random
import string
import random
from Crypto.Random import get_random_bytes
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import binomtest
import ast
import re
from Crypto.Util import Counter

#---Settings--------------------------------------------------------------------
model_path="/content/81-7-2-MLP.h5"
scriptnr = "32-2"
mode = "CTR"
p_param = "nr-lett-8-By"
k_param = "8-By"
datasets = 50000
extras = "p-RND-2"

file_name = f"{scriptnr}_{mode}_p-{p_param}_k-{k_param}_{datasets}_{extras}.csv"
folder='/content'
csv_path = os.path.join(folder, file_name)

K_LEN_BIT = 64
P_LEN = 8
#same k, p speziell-> unter create csv anpassen

#-2--Generiere neuen Datensatz--------------------------------------------------

def generate_des_key():
    bit_length = K_LEN_BIT
    max_val = 2 ** bit_length - 1
    rand_int = random.randint(0, max_val)
    byte_len = (bit_length + 7) // 8
    rand_bytes = rand_int.to_bytes(byte_len, byteorder='big')
    padded_bytes = rand_bytes.rjust(8, b'\x00')
    return padded_bytes

def generate_p_txt_bytes():
    characters = string.ascii_letters + string.digits
    kombi = ''.join(random.choices(characters, k=8))
    b = kombi.encode('utf-8')
    assert len(b) == 8, f"Plaintext bytes length ist {len(b)}, sollte 8 sein!"
    return b

#-Klartext komplett zufällig
def generate_rnd_p_txt_bytes():
    b = os.urandom(8)
    assert len(b) == 8, f"Plaintext bytes length ist {len(b)}, sollte 8 sein!"
    return b

#-Klartext=nur Buchstaben
def generate_letters_p_txt_bytes():
    characters = string.ascii_letters
    kombi = ''.join(random.choices(characters, k=8))
    b = kombi.encode('utf-8')
    assert len(b) == 8, f"Plaintext bytes length ist {len(b)}, sollte 8 sein!"
    return b

#-Klartext=nur Kleinbuchstaben
def generate_letters_upper_p_txt_bytes():
    characters = string.ascii_lowercase
    kombi = ''.join(random.choices(characters, k=8))
    b = kombi.encode('utf-8')
    assert len(b) == 8, f"Plaintext bytes length ist {len(b)}, sollte 8 sein!"
    return b

def generate_letters_upper_a_k_p_txt_bytes():
    characters = 'abcdefghijk'  # nur kleine Buchstaben von a bis k
    kombi = ''.join(random.choices(characters, k=8))
    b = kombi.encode('utf-8')
    assert len(b) == 8, f"Plaintext bytes length ist {len(b)}, sollte 8 sein!"
    return b

def generate_letters_upper_a_d_p_txt_bytes():
    characters = 'abcd'  # nur kleine Buchstaben von a bis c
    kombi = ''.join(random.choices(characters, k=8))
    b = kombi.encode('utf-8')
    assert len(b) == 8, f"Plaintext bytes length ist {len(b)}, sollte 8 sein!"
    return b

def encrypt_with_des_ECB(key, data_bytes):
    cipher = DES.new(key, DES.MODE_ECB)
    encrypted_data = cipher.encrypt(data_bytes)
    assert len(encrypted_data) == 8, f"Ciphertext Länge ist {len(encrypted_data)}, sollte 8 Bytes sein"
    return encrypted_data.hex()

def encrypt_with_des_CTR(key, data, ctr):
    cipher = DES.new(key, DES.MODE_CTR, counter=ctr)
    encrypted_data = cipher.encrypt(data)
    assert len(encrypted_data) == 8, f"Ciphertext Länge ist {len(encrypted_data)} und sollte 8 Bytes sein!"
    result = encrypted_data.hex()
    return result

#-Datensatzparamerter ggf. hier setzen
def create_csv():
   # key = generate_des_key() # wenn k=same

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['p', 'c', 'k'])

        for i in range(datasets):
            #plaintext_bytes = generate_p_txt_bytes()
            plaintext_bytes = generate_rnd_p_txt_bytes()
            #plaintext_bytes = generate_letters_p_txt_bytes()
            #plaintext_bytes = generate_letters_upper_p_txt_bytes()
            #plaintext_bytes = generate_letters_upper_a_d_p_txt_bytes()

            key = generate_des_key() # diff. keys

            if mode == "ECB":
                cipher_hex = encrypt_with_des_ECB(key, plaintext_bytes)
            else:
              nonce = get_random_bytes(4)
              ctr=Counter.new(32,prefix=nonce, initial_value=i)
              cipher_hex = encrypt_with_des_CTR(key, plaintext_bytes, ctr)

            writer.writerow([plaintext_bytes.hex(), cipher_hex, key.hex()])

if __name__ == '__main__':
    create_csv()
    print("CSV-Datei wurde erfolgreich erstellt.")

#-3--Lade Model, starte Vorhersage----------------------------------------------
path_test_data=csv_path

BIT_LENGTH = 64

#-Modell und Daten laden
model = load_model(model_path)
test_df = pd.read_csv(path_test_data)

mn=dateiname = os.path.basename(model_path)
prefix=f"M-{mn}__D-{mode}-{datasets}-{extras}"
results_data_file=f"{prefix}-PREDICTION.csv"

#-Hilfsfunktionen---------------------------------------------------------------
#-Datenvorbereitung
def hex_to_bin_vector(hex_str, length=BIT_LENGTH):#problem beim p=zahlen
    bin_str = bin(int(hex_str, 16))[2:].zfill(length)
    return np.array([int(b) for b in bin_str])

def binvec_to_hex(binvec):
    bits_str = ''.join(str(int(b)) for b in binvec)
    return hex(int(bits_str, 2))[2:].zfill(len(binvec) // 4)

def extract_features_from_c_advanced(hex_str, bit_length=BIT_LENGTH):
    bin_vec = hex_to_bin_vector(hex_str, bit_length)
    features = []

    #-1-Roh-Bits
    features.extend(bin_vec)

    #-2-Popcount & Parität
    popcount = np.sum(bin_vec)
    features.append(popcount / bit_length)
    features.append(popcount % 2)

    #-3-Quartile (lokale Popcounts)
    q = bit_length // 4
    for i in range(4):
        features.append(np.sum(bin_vec[i*q:(i+1)*q]) / q)

    #-4-Bit-Flips (0<->1)
    bit_flips = sum(bin_vec[i] != bin_vec[i+1] for i in range(bit_length - 1))
    features.append(bit_flips / (bit_length - 1))

    #-5-XOR benachbarter Bits
    for i in range(bit_length - 1):
        features.append(bin_vec[i] ^ bin_vec[i+1])

    #-6-AND benachbarter Bits
    for i in range(bit_length - 1):
        features.append(bin_vec[i] & bin_vec[i+1])

    #-7-S-Box Pattern: 4-Bit Nibbles
    for i in range(0, bit_length, 4):
        nibble = bin_vec[i:i+4]
        if len(nibble) == 4:
            index = int("".join(str(b) for b in nibble), 2)
            features.append(index / 15)

    #-8-Shifted Inputs (zirkular)
    features.extend(np.roll(bin_vec, -1))  # Linksrotation
    features.extend(np.roll(bin_vec, 1))   # Rechtsrotation

    return np.array(features)

#-4---Vorhersage----------------------------------------------------------------
X_test_c = np.vstack(test_df['c'].apply(extract_features_from_c_advanced).values)
X_test_p = np.vstack(test_df['p'].apply(hex_to_bin_vector).values)
X_test_k = np.vstack(test_df['k'].apply(hex_to_bin_vector).values)

X_test = np.hstack([X_test_c])

y_pred_p, y_pred_k = model.predict(X_test)

#-Binarisieren
y_pred_p_bin = (y_pred_p > 0.5).astype(int)
y_pred_k_bin = (y_pred_k > 0.5).astype(int)

# ------------------------------------------------------------------------------
results = []
#-Vergleich & Ausgabe
for i in range(len(test_df)):
    p_acc = np.mean(y_pred_p_bin[i] == X_test_p[i]) * 100
    k_acc = np.mean(y_pred_k_bin[i] == X_test_k[i]) * 100
    #print(p_acc, k_acc)

    results.append({
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

results_df = pd.DataFrame(results)
results_df.head()

results_df.to_csv(results_data_file, index=False)
files.download(results_data_file)
print("Erstellt val-results")

'''
print(f"Sample {i+1}:")
print(f"  p: true={true_p}, pred={pred_p}, accuracy={p_acc:.1f}%")
print(f"  k: true={true_k}, pred={pred_k}, accuracy={k_acc:.1f}%\
'''

#-5---Evaluation----------------------------------------------------------------
from collections import Counter

prefix=f"M-{mn}__D-{mode}-{datasets}-{extras}"
predictions_file_path=results_data_file

#-CSV-Datei einlesen
df = pd.read_csv(predictions_file_path)
#-------------------------------------------------------------------------------
valid_results_file_name=f"{prefix}__RESULT.txt"

#-Spalten->Variabe
p_true_bin='p_true_bin'
p_pred_bin='p_pred_bin'

k_true_bin='k_true_bin'
k_pred_bin='k_pred_bin'

p_true_hex='p_true'
p_pred_hex='p_pred'

k_true_hex='k_true'
k_pred_hex='k_pred'

#-3---Hilfsfunktionen-----------------------------------------------------------

def parse_binary_list_string(list_string):
    # Entferne Klammer the brackets and split by spaces
    return [int(x) for x in list_string.strip("[]").split() if x in {'0', '1'}]

#-Genauigkeut der gesamt- p/k min, max, Durchschnitt
def p_max_aver_min_accuracy():
    #-Falls p_acc als String gelesen wurde, in float umwandeln
    df['p_acc'] = df['p_acc'].astype(float)

    #-Minimum, Maximum und Mittelwert über alle Zeilen
    min_acc = df['p_acc'].min()
    mean_acc = df['p_acc'].mean()
    max_acc = df['p_acc'].max()

    #-Ausgabe
    p_min_acc=f"p min acc: {min_acc}"
    p_mitt_acc=f"p mitt acc: {mean_acc}"
    p_max_acc=f"p max acc: {max_acc}"

    text=""
    text += f"- P - {p_min_acc}\n"
    text += f"- P - {p_mitt_acc}\n"
    text += f"- P - {p_max_acc}\n"
    text += f"\n{'-'*40}\n"
    print(f"{p_min_acc} \n {p_mitt_acc} \n {p_max_acc}")

    return text

def k_max_aver_min_accuracy():
    #-Falls p_acc als String gelesen wurde, in float umwandeln

    df['k_acc'] = df['k_acc'].astype(float)

    #-Minimum, Maximum und Mittelwert über alle Zeilen
    min_acc_ = df['k_acc'].min()
    mean_acc_ = df['k_acc'].mean()
    max_acc_ = df['k_acc'].max()

    #-Ausgabe
    k_min_acc=f"k min acc: {min_acc_}"
    k_mitt_acc=f"k mitt acc: {mean_acc_}"
    k_max_acc=f"k max acc: {max_acc_}"

    text=""
    text += f"- K - {k_min_acc}\n"
    text += f"- K - {k_mitt_acc}\n"
    text += f"- K - {k_max_acc}\n"
    text += f"\n{'-'*40}\n"
    print(f"{k_min_acc} \n {k_mitt_acc} \n {k_max_acc}")

    return text

#-Berechnungsfunktionen
def get_info(true_bin, pred_bin, test_name):

  text = ""

  param_1=true_bin
  param_2=pred_bin

  df[param_1] = df[param_1].apply(parse_binary_list_string)
  df[param_2] = df[param_2].apply(parse_binary_list_string)

  #-Alle Werte als eine Liste zusammenfügen
  y_true = np.concatenate(df[param_1].values)
  y_pred = np.concatenate(df[param_2].values)

  #---Metriken berechnen
  accuracy = accuracy_score(y_true, y_pred)
  precision = precision_score(y_true, y_pred)
  recall = recall_score(y_true, y_pred)
  f1 = f1_score(y_true, y_pred)
  roc_auc = roc_auc_score(y_true, y_pred)

  #---Binomialtest
  correct = np.sum(y_true == y_pred)
  total = len(y_true)
  p_value = binomtest(correct, total, p=0.5, alternative='greater').pvalue

  #---Ausgabe
  print(f"Ergebnisse der Vorhersageanalyse: {param_1} <-> {param_2}")
  print(f" Accuracy:      {accuracy:.3f}")
  print(f" Precision:     {precision:.3f}")
  print(f" Recall:        {recall:.3f}")
  print(f" F1-Score:      {f1:.3f}")
  print(f" ROC-AUC:       {roc_auc:.3f}")
  print(f" Treffer:       {correct} von {total}")
  print(f" p-Wert:        {p_value:.5f}  → {'Signifikant' if p_value < 0.05 else 'Nicht signifikant'}")


  #---Estelle Text---
  text += f"{test_name}\n\n"
  text += f" Accuracy:      {accuracy:.3f}\n"
  text += f" Precision:     {precision:.3f}\n"
  text += f" Recall:        {recall:.3f}\n"
  text += f" F1-Score:      {f1:.3f}\n"
  text += f" ROC-AUC:       {roc_auc:.3f}\n"
  text += f" Treffer:       {correct} von {total}\n"
  text += f" p-Wert:        {p_value:.5f}  → {'Signifikant' if p_value < 0.05 else 'Nicht signifikant'}\n"
  text += f"\n{'-'*40}\n"
  return text

#-Wiederholende Elemente-min,max,Anzahl an Wiederholungen
def analyze_column_duplicates(series, name):
    # Liste zu Tupel konvertieren fürs Zählen
    tuples = [tuple(x) if isinstance(x, list) else x for x in series]
    counter = Counter(tuples)

    text=f"{name}\n"

    if all(count == 1 for count in counter.values()):

        text += f"Keine Wiederholende Werte \n"
        text += f"\n{'-'*40}\n"

        return text

    num_unique = len(counter)
    min_count = min(counter.values())
    max_count = max(counter.values())
    min_values = [val for val, count in counter.items() if count == min_count]
    max_values = [val for val, count in counter.items() if count == max_count]



    text += f"Anzahl der Einzelwerte: {num_unique}\n"
    text += f"Wert: {', '.join(map(str, min_values))} wiederholt sich {min_count}  Mal \n"
    text += f"Wert: {', '.join(map(str, max_values))} wiederholt sich {max_count} Mal \n"
    text += f"\n{'-'*40}\n"

    return text

#-4---Ergebnisse zusammenfassen-----------------------------------------------

summary = f"Validation Ergebnisse\n{'-'*40}\n"

summary += p_max_aver_min_accuracy()
summary += k_max_aver_min_accuracy()
summary += f"\n{'-'*40}\n"

test_name=f"Ergebnisse der Vorhersageanalyse: {p_true_bin} <-> {p_pred_bin}"
summary += get_info(p_true_bin, p_pred_bin, test_name)

test_name=f"Ergebnisse der Vorhersageanalyse: {k_true_bin} <-> {k_pred_bin}"
summary += get_info(k_true_bin, k_pred_bin, test_name)

summary += analyze_column_duplicates(df[p_true_hex],"p_true_hex")
summary += analyze_column_duplicates(df[p_pred_hex],"p_pred_hex")
summary += analyze_column_duplicates(df[k_true_hex],"k_true_hex")
summary += analyze_column_duplicates(df[k_pred_hex],"k_pred_hex")

#-Speichern
with open(valid_results_file_name, 'w') as f:
      f.write(summary)

#-Download
files.download(valid_results_file_name)