import csv
from Crypto.Cipher import DES
from Crypto.Util.Padding import pad
import os
from Crypto.Random import get_random_bytes
import random
import string
from Crypto.Util import Counter

#---Settings-------------------------
scriptnr = "84F"
mode = "CTR"
p_param = "nr-lett-8-By"
k_param = "same_8-By"
datasets = 1000
extras = ""
file_name = f"{scriptnr}_{mode}_p-{p_param}_k-{k_param}_{datasets}_{extras}.csv"
folder='/home/pc/Schreibtisch/Project/Data/'
csv_path = os.path.join(folder, file_name)

K_LEN_BIT = 64
P_LEN = 8
samekey=get_random_bytes(8) 
#-------------------------------------
def generate_p_txt_bytes():
    characters = string.ascii_letters + string.digits
    kombi = ''.join(random.choices(characters, k=8))  # 8 Zeichen
    b = kombi.encode('utf-8')
    assert len(b) == 8, f"Plaintext bytes length ist {len(b)}, sollte 8 sein!"
    return b
 
def encrypt_with_des(key, data, ctr):
    cipher = DES.new(key, DES.MODE_CTR, counter=ctr)
    encrypted_data = cipher.encrypt(data)  
    assert len(encrypted_data) == 8, f"Ciphertext Länge ist {len(encrypted_data)}, sollte 8 Bytes sein"
    result = encrypted_data.hex()
    return result
 
def create_csv():
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['p', 'c', 'k'])

        for i in range(datasets):
            plaintext_bytes = generate_p_txt_bytes()
            nonce = get_random_bytes(4)
            key = samekey
        
            ctr=Counter.new(32,prefix=nonce, initial_value=i)

            cipher = encrypt_with_des(key, plaintext_bytes, ctr)
            #print (cipher)

            writer.writerow([plaintext_bytes.hex(), cipher, key.hex()])

if __name__ == '__main__':
    create_csv()
    print("CSV-Datei wurde erfolgreich erstellt.")
