import os

SOURCE = r"C:\Users\follo\OneDrive\Desktop\test"
TEST = r"C:\Users\follo\OneDrive\Desktop\test\vid1"
DESTINATION = r"C:\Users\follo\OneDrive\Desktop\test\gest"


lsit = [5,3,1,8]
sublsit = lsit[0:3]

print(sublsit)






videotest = ["vid1","vid2", "vid3"]

for vid in videotest: #per ogni label (per ogni gesto)
    try:
        os.makedirs(os.path.join(SOURCE, vid))
    except:
        pass #se esistono già, skippa la creazione


gestures = ["ciao", "bella", "zio"]

for label in gestures: #per ogni label (per ogni gesto)
    try:
        os.makedirs(os.path.join(DESTINATION, label))
    except:
        pass #se esistono già, skippa la creazione


allvid = os.listdir(SOURCE) 

# iterate on all files to move them to destination folder
for vid in allvid:
    if not vid == "gest":       
        src_path = os.path.join(SOURCE, vid)
        dst_path = os.path.join(DESTINATION, label, vid)
        os.rename(src_path, dst_path)

