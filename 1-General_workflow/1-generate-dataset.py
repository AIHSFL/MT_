'''
Script (vereinigt 81-1-81-9)für Erstellung der Trainingsdatensätze
Ablauf:
1. Einstellungen vornehmen
2. Daten generieren
3. Datensatz [c][p][k] wird als eine CSV Datei erstellt.
'''

import csv
from Crypto.Cipher import DES
from Crypto.Util.Padding import pad
import os
from Crypto.Random import get_random_bytes
import random
import string
from Crypto.Util import Counter

#---Settings-------------------------
scriptnr = "88F"
mode = "CTR"
p_param = "nr-lett-8-By"
k_param = "8-By"
datasets = 2000000
extras = ""
file_name = f"{scriptnr}_{mode}_p-{p_param}_k-{k_param}_{datasets}_{extras}.csv"
folder='/home/pc/Schreibtisch/Project/Data/'
csv_path = os.path.join(folder, file_name)

#-weitere Settings unter create_csv

K_LEN_BIT = 64
P_LEN = 8

#-------------------------------------
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
    kombi = ''.join(random.choices(characters, k=8))  # 8 Zeichen
    b = kombi.encode('utf-8')
    assert len(b) == 8, f"Plaintext bytes length ist {len(b)}, sollte 8 sein!"
    return b

def generate_rnd_p_txt_bytes():
    b = os.urandom(8)  # Gibt ein Bytes-Objekt mit 8 zufälligen Bytes zurück
    assert len(b) == 8, f"Plaintext bytes length ist {len(b)}, sollte 8 sein!"
    return b

def generate_letters_p_txt_bytes():
    characters = string.ascii_letters 
    kombi = ''.join(random.choices(characters, k=8))  # 8 Zeichen
    b = kombi.encode('utf-8')
    assert len(b) == 8, f"Plaintext bytes length ist {len(b)}, sollte 8 sein!"
    return b

def generate_letters_upper_p_txt_bytes():
    characters = string.ascii_lowercase 
    kombi = ''.join(random.choices(characters, k=8)) 
    b = kombi.encode('utf-8')
    assert len(b) == 8, f"Plaintext bytes length ist {len(b)}, sollte 8 sein!"
    return b

def encrypt_with_des_CTR(key, data, ctr):
    cipher = DES.new(key, DES.MODE_CTR, counter=ctr)
    encrypted_data = cipher.encrypt(data) 
    assert len(encrypted_data) == 8, f"Ciphertext Länge ist {len(encrypted_data)}, sollte 8 Bytes sein"
    result = encrypted_data.hex()
    return result
 
def encrypt_with_des_ECB(key, data_bytes):
    cipher = DES.new(key, DES.MODE_ECB)
    encrypted_data = cipher.encrypt(data_bytes)
    assert len(encrypted_data) == 8, f"Ciphertext Länge ist {len(encrypted_data)}, sollte 8 Bytes sein"
    return encrypted_data.hex()

def create_csv():
    #-Settings-------------
    #plaintext_bytes = generate_rnd_p_txt_bytes()#if p==same
    #key = generate_des_key() #if k==same
    #----------------------

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['p', 'c', 'k'])

        for i in range(datasets):
            
            #-Settings-------------
            key = generate_des_key() # k=!same
            plaintext_bytes = generate_letters_upper_p_txt_bytes() # p=!same
            #----------------------
            
            if mode=="ECB":
                cipher_hex = encrypt_with_des_ECB(key, plaintext_bytes)
            else:
                nonce = get_random_bytes(4)
                ctr=Counter.new(32,prefix=nonce, initial_value=i)
                cipher_hex = encrypt_with_des_CTR(key, plaintext_bytes, ctr)

            writer.writerow([plaintext_bytes.hex(), cipher_hex, key.hex()])

if __name__ == '__main__':
    create_csv()
    print("CSV-Datei wurde erfolgreich erstellt.")
