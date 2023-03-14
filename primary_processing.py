import json
from chord_extractor.extractors import Chordino
import yadisk


def fun (song_wav):  # program for processing 
    chordino = Chordino()
    chords = chordino.extract(song_wav)
    data = []
    prev_len = 0

    for chord in chords:
        data.append(chord[0])
        data.append(chord[1] - prev_len)
        prev_len = chord[1]

    return data  


yandex = yadisk.YaDisk(token="some token")

lst = list(yandex.listdir("path on the yandex disk"))


for i in range(len(lst)):  # downloading
    with open('path on the yandex disk'+lst[i]["name"], 'wb+') as f:
        print('path on the yandex disk'+lst[i]["name"])
        yandex.download('path on the yandex disk'+lst[i]["name"], f) 

data = {}

for i in range(len(lst)):  # processing 
    print(i)
    data[lst[i]['name']] = fun('path on the computer'+lst[i]["name"])

with open('data2.json', 'w') as f:  # writing to the dataset
    json.dump(data, f)
