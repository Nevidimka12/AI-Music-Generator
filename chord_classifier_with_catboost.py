!pip install catboost
import json
import os
import numpy as np
import pandas as pd
from catboost.utils import create_cd
from catboost import CatBoostClassifier
from catboost import CatBoostRegressor

np.set_printoptions(precision=4)

df_for_chord = pd.read_csv('/content/drive/MyDrive/DataFrame.csv', index_col = "Unnamed: 0")
df_for_duration = pd.read_csv('/content/drive/MyDrive/DataFrame.csv', index_col = "Unnamed: 0")
df_for_chord = df_for_chord.drop('99', axis = 1)

from sklearn.model_selection import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(df_for_chord.drop('target', axis = 1), df_for_chord['target'], random_state=13, test_size=0.2)
X_train1, X_validation1, y_train1, y_validation1 = train_test_split(df_for_duration.drop('99', axis = 1), df_for_duration['99'], random_state=13, test_size=0.2)

feature_names = dict()
for column, name in enumerate(df_for_chord):
    if column == 0:
        continue
    feature_names[column] = name

create_cd(
    label=0,
    cat_features=list(range(1, df_for_chord.columns.shape[0])),
    feature_names=feature_names,
    output_path=os.path.join('train.cd')
)

model = CatBoostClassifier(
    iterations=300,
    depth = 5,
    learning_rate=0.4,
    task_type = "GPU"
)

model.fit(
    X_train, y_train,
    eval_set=(X_validation, y_validation),
    plot=True
)

model2 = CatBoostRegressor(iterations=600,
                        learning_rate=0.01,
                        task_type = "GPU"
                        )
model2.fit(
    X_train1, y_train1,
    eval_set=(X_validation1, y_validation1),
    plot=True
)

def filler(arr):
    for_model = []
    while(len(for_model) < 98):
        for_model += arr
    return for_model[:98]


def decode_chord_data(chord_data):
    file = open("/content/drive/MyDrive/Decode.json")
    decoder = json.load(file)
    file.close()

    res = []
    for chord, duration in chord_data:
        res.append((decoder[str(chord)], duration))

    return res


def encode_chord_data(chord_data):

    file = open("/content/drive/MyDrive/Decode.json")
    decoder = json.load(file)
    file.close()

    encoder = dict()
    for key, value in decoder.items():
        encoder[value] = int(key)

    res = []
    for chord, duration in chord_data:
        res.append((encoder[chord], duration))

    return res


def predict_sequence(chord_data, n, predicted_chord_durations=1):
    chord_data = encode_chord_data(chord_data)
    print(chord_data)
    sequence = [j for sub in chord_data for j in sub]
    result = list(chord_data)

    if len(sequence) > 98:
        sequence = sequence[-98:]
    elif len(sequence) < 98:
        sequence = filler(sequence)
    for _ in range(n):
        predicted_chord = int(model.predict(data=sequence))
        sequence.append(predicted_chord)
        predicted_dur = model2.predict(data=sequence)
        sequence.append(predicted_dur)
        sequence = sequence[2:]

        result.append((sequence[-2], sequence[-1]))

    return decode_chord_data(result)
