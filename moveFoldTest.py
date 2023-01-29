import os

DATADIR = r"C:\Users\follo\OneDrive\Desktop\test"

gestures = ["ciao", "bella", "zio"]

for label in gestures: #per ogni label (per ogni gesto)
    try:
        os.makedirs(os.path.join(DATADIR, label))
    except:
        pass #se esistono già, skippa la creazione