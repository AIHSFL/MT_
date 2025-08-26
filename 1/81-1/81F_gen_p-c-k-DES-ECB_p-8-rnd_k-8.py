import csv
from Crypto.Cipher import DES
from Crypto.Util.Padding import pad
import os
import random
import string

#---Settings-------------------------
scriptnr = "81F"
mode = "ECB"
p_param = "nr-lett-8-By"
k_param = "8-By"
datasets = 1000
extras = ""
file_name = f"{scriptnr}_{mode}_p-{p_param}_k-{k_param}_{datasets}_{extras}.csv"
folder='/home/pc/Schreibtisch/Project/Data'
csv_path = os.path.join(folder, file_name)

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

def encrypt_with_des(key, data_bytes):
    cipher = DES.new(key, DES.MODE_ECB)
    encrypted_data = cipher.encrypt(data_bytes)  # kein Padding!
    assert len(encrypted_data) == 8, f"Ciphertext Länge ist {len(encrypted_data)}, sollte 8 Bytes sein"
    return encrypted_data.hex()

def create_csv():

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['p', 'c', 'k'])
        for _ in range(datasets):
            plaintext_bytes = generate_p_txt_bytes()
            key = generate_des_key()
            cipher_hex = encrypt_with_des(key, plaintext_bytes)
            writer.writerow([plaintext_bytes.hex(), cipher_hex, key.hex()])

if __name__ == '__main__':
    create_csv()
    print("CSV-Datei wurde erfolgreich erstellt.")
