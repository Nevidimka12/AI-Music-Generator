!pip install catboost
import json
import os
import numpy as np
import pandas as pd
from catboost.utils import create_cd
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV

np.set_printoptions(precision=4)

df = pd.read_csv('/content/DataFrame.csv', index_col = "Unnamed: 0")
df = df.drop('99', axis = 1)

from sklearn.model_selection import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(df.drop('target', axis = 1), df['target'], random_state=13, test_size=0.2)

feature_names = dict()
for column, name in enumerate(df):
    if column == 0:
        continue
    feature_names[column] = name

create_cd(
    label=0,
    cat_features=list(range(1, df.columns.shape[0])),
    feature_names=feature_names,
    output_path=os.path.join('train.cd')
)

model = CatBoostClassifier(
    iterations=100,
    learning_rate=1,
    task_type = "GPU"
)
model.fit(
    X_train, y_train,
    eval_set=(X_validation, y_validation),
    plot=False
)
parameters = {'depth'         : [4,5],'learning_rate' : [0.4, 0.5, 0.6], 'iterations': [200, 250, 300]
                 }

Grid_model = GridSearchCV(estimator=model, param_grid = parameters, cv = 2, n_jobs=-1)
Grid_model.fit(X_train, y_train)

print(" Results from Grid Search " )
print("\n The best estimator across ALL searched params:\n",Grid_model.best_estimator_)
print("\n The best score across ALL searched params:\n",Grid_model.best_score_)
print("\n The best parameters across ALL searched params:\n",Grid_model.best_params_)

def filler(arr):
    for_model = []
    while(len(for_model) < 98):
        for_model += arr
    return for_model[:98]


def decode_chord_data(chord_data):
    file = open("/content/Decode.json")
    decoder = json.load(file)
    file.close()

    res = []
    for chord, duration in chord_data:
        res.append((decoder[str(chord)], duration))

    return res


def encode_chord_data(chord_data):

    file = open("/content/Decode.json")
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
    sequence = [j for sub in chord_data for j in sub]
    result = list(chord_data)

    if len(sequence) > 98:
        sequence = sequence[-98:]
    elif len(sequence) < 98:
        sequence = filler(sequence)
    for _ in range(n):
        predicted_chord = int(model.predict(data=sequence))
        predicted_dur = predicted_chord_durations

        sequence.append(predicted_chord)
        sequence.append(predicted_dur)
        sequence = sequence[2:]

        result.append((sequence[-2], sequence[-1]))

    return decode_chord_data(result)
