# -*- coding: utf-8 -*-

'''
Auswertung der Vorhersagen.
Es wird eine CSV-Datei aus Modell-Training-Script eingelesen und ausgewertet.
Das Ergebnis wird in einer Textdatei zusammengefasst.

Ablauf:
1. Vorhersageergebnisse im Format 
[c]
[p_true_hex][p_predicted_hex][p_true_binär][p_predicted_binär][p_prediction_accuracy]
[k_true_hex][k_predicted_hex][k_true_binär][k_predicted_binär][k_prediction_accuracy]

werden als CSV Datei eingelesen.

2. Es wird minimale, maximale und durchschnittliche Accuracy über alle Datenzeilen
für k und p ermittelt

3. Es wird Gesamt-Accuracy durch Bits-Übereinstimmungsabgleich ermittelt.

4. Es wird die Trefferanzahl sowie die Signifikanz der Übereinstimmungen ermittelt

5. Es gibt weitere experimentelle Tests (nicht verwendet).

6. Ergebnisse werden in einer Textdatei zusammengefasst.

'''

#-1--Imports-----------------------------------------------------------
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import binomtest
import numpy as np
import ast
import re
import os
from google.colab import files
from collections import Counter

#-2--Settings---------------------------------------------------------
prefix="81-7-2-statist-train-daten"
file_path="/content/81-7-2-prediction_on_train-ds.csv"

#-------------------------------
valid_results_file_name=f"{prefix}__results.txt"

#-Spalten->Variabe
p_true_bin='p_true_bin'
p_pred_bin='p_pred_bin'

k_true_bin='k_true_bin'
k_pred_bin='k_pred_bin'

p_true_hex='p_true'
p_pred_hex='p_pred'

k_true_hex='k_true'
k_pred_hex='k_pred'

#-CSV-Datei einlesen
df = pd.read_csv(file_path)

#---------------------------------------------------------------------
#-3--Hilfsfunktionen---

def parse_binary_list_string(list_string):
    #-Entferne Klammer
    return [int(x) for x in list_string.strip("[]").split() if x in {'0', '1'}]

#-Genauigkeut der gesamt-p/k min, max, Durchschnitt
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

  #-Alle Werte als eine Liste zusammenfügen-mehrere Zeilen
  y_true = np.concatenate(df[param_1].values)
  y_pred = np.concatenate(df[param_2].values)

  #-Metriken berechnen---
  accuracy = accuracy_score(y_true, y_pred)
  precision = precision_score(y_true, y_pred)
  recall = recall_score(y_true, y_pred)
  f1 = f1_score(y_true, y_pred)
  roc_auc = roc_auc_score(y_true, y_pred)

  #---Binomialtest---
  correct = np.sum(y_true == y_pred)
  total = len(y_true)
  p_value = binomtest(correct, total, p=0.5, alternative='greater').pvalue

  #---Ausgabe---
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

#-4---Ergebnisse zusammenfassen--------------------------

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