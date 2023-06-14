import json
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
import collections
import numpy as np
import pandas as pd
import pretty_midi
import tensorflow as tf
import librosa
import melody_classifier as mc
from random import shuffle

# model = CatBoostClassifier()
# model.load_model("Chord_model_filt")
model_chords = CatBoostClassifier()
model_chords.load_model('cat_chords')


def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
    mse = (y_true - y_pred) ** 2
    positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
    return tf.reduce_mean(mse + positive_pressure)


tf.keras.utils.get_custom_objects()['mse_with_positive_pressure'] = mse_with_positive_pressure
model_melody = tf.keras.models.load_model('melodyLSTM13.06.23.h5')

model = tf.keras.models.load_model('chords.h5')


vocab_size = 128
seq_length = 25


def predict(model, input_seq):
    model.eval()
    with torch.no_grad():
        predictions = model(input_seq)
    return predictions


def filler(arr):
    for_model = []
    while(len(for_model) < 8):
        for_model += arr
    for_model = for_model[len(for_model)-9:len(for_model)-1]
    return for_model


def shuffle_for_pair(arr):
    new_arr = []
    for i in range(0, len(arr)-1, 2):
        new_arr.append((arr[i], arr[i+1]))
    shuffle(new_arr)
    y = []
    for i in new_arr:
        y.append(i[0])
        y.append(i[1])
    return y


def filler_for_chord(arr):
    for_model = []
    while(len(for_model) < 98):
        for_model += arr
    return for_model[:98]


def predict_sequence(chord_data, pred_len):
    chord_data = encode_chord_data(chord_data)
    sequence_for_chord = [i for p in chord_data for i in p]
    sequence_for_duration = [i for p in chord_data for i in p]
    result = list(chord_data)

    seq_dur = []
    if len(sequence_for_duration) > 8:
        sequence_for_duration = sequence_for_duration[len(sequence_for_duration)-9:len(sequence_for_duration)-1]
    elif len(sequence_for_duration) < 8:
        sequence_for_duration = filler(sequence_for_duration)
    for i in range(0, len(sequence_for_duration)-1, 2):
        chord = sequence_for_duration[i]
        duration = sequence_for_duration[i+1]
        seq_dur.append(chord)
        seq_dur.append(duration)

    if len(sequence_for_chord) > 98:
        sequence_for_chord = sequence_for_chord[-98:]
    elif len(sequence_for_chord) < 98:
        sequence_for_chord = filler_for_chord(sequence_for_chord)

    now_len = 0

    while now_len < pred_len:
        predicted_chord = int(model_chords.predict(data=sequence_for_chord))
        seq_dur.append(predicted_chord)
        seq_for_predict = convert(seq_dur)
        predicted_duration = model.predict(seq_for_predict)[0][0]
        seq_dur.append(predicted_duration)
        sequence_for_chord.append(predicted_chord)
        sequence_for_chord.append(predicted_duration)
        sequence_for_chord = sequence_for_chord[2:]
        seq_dur = seq_dur[2:]
        result.append((predicted_chord, predicted_duration))
        now_len+=predicted_duration

    return decode_chord_data(result)


def decode_chord_data(chord_data):
    file = open("Decode.json")
    decoder = json.load(file)
    file.close()

    res = []
    for chord, duration in chord_data:
        res.append((decoder[str(int(chord))], duration))

    return res


def encode_chord_data(chord_data):
    file = open("Decode.json")
    decoder = json.load(file)
    file.close()

    encoder = dict()
    for key, value in decoder.items():
        encoder[value] = int(key)

    res = []
    for chord, duration in chord_data:
        res.append((encoder[chord], duration))

    return res


def cut_out_silence(chord_data, silence_duration):  # Убирает тишину в конце предсказанной песни

    if chord_data[-1][0] == 'N':
        new_chord_data = list([e for e in chord_data[:-1]])
        new_chord_data.append((chord_data[-1][0], silence_duration))
    else:
        new_chord_data = chord_data

    return new_chord_data


def convert(arr):
    new_arr = np.zeros((1, 9, 1))
    for i in range(0, len(arr)//2-1, 2):
        new_arr[0][i][0] = arr[i]
        new_arr[0][i][0] = arr[i+1]
    new_arr[0][8][0] = arr[8]
    return new_arr


def add_N(data: pd.DataFrame) -> list:
    print(data.columns)
    res = []
    data = data.to_numpy()
    for i in range(1, len((data))):
        res.append((librosa.midi_to_note(data[i][0]), float(data[i][2])))
        if data[i-1][4] != data[i][3]:
            res.append(('N', float(data[i][3]) - float(data[i-1][4])))
    res.append((librosa.midi_to_note(data[-1][0]), float(data[-1][2])))
    return res


def add_time_intervals(notes):
    new_notes = []
    previous_start_time = 0.0

    for note, duration in notes:
        start_time = previous_start_time
        end_time = start_time + duration
        new_notes.append([note, duration, start_time, end_time])
        previous_start_time = end_time
    return new_notes


def remove_duplicates(a):
    ans = []
    dur = 0
    for i in range(len(a)-1):
        dur += a[i][1]
        current_note = a[i][0]
        next_note = a[i + 1][0]
        if current_note != next_note:
            ans.append((current_note, dur))
            dur = 0
    ans.append((a[-1][0], a[-1][1]+dur))

    return ans


def merge_data(t):
    data = collections.defaultdict(list)
    prev_start_time = 0.0
    for i in t:
        if i[0] != 'N':
            data['pitch'].append(i[0])
            data['start'].append(i[2])
            data['end'].append(i[3])
            data['step'].append(i[2] - prev_start_time)
            data['duration'].append(i[1])
            prev_start_time = i[2]
    return data


def predict_next_note(notes: np.ndarray, keras_model: tf.keras.Model, temperature: float = 1.0) -> int:
    assert temperature > 0
    inputs = tf.expand_dims(notes, 0)
    predictions = keras_model.predict(inputs)
    pitch_logits = predictions['pitch']
    step = predictions['step']
    duration = predictions['duration']
    pitch_logits /= temperature
    pitch = tf.random.categorical(pitch_logits, num_samples=1)
    pitch = tf.squeeze(pitch, axis=-1)
    duration = tf.squeeze(duration, axis=-1)
    step = tf.squeeze(step, axis=-1)
    step = tf.maximum(0, step)
    duration = tf.maximum(0, duration)
    return int(pitch), float(step), float(duration)


def notes_to_midi(
  notes: pd.DataFrame,
  out_file: str,
  instrument_name: str,
  velocity: int = 100,  # note loudness
) -> pretty_midi.PrettyMIDI:

    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program(instrument_name))

    prev_start = 0
    for i, note in notes.iterrows():
        start = float(prev_start + note['step'])
        end = float(start + note['duration'])
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=int(note['pitch']),
            start=start,
            end=end,
        )
        instrument.notes.append(note)
        prev_start = start

    pm.instruments.append(instrument)
    pm.write(out_file)
    return pm


def save_predict_melody(filename, dur):
    test = collections.defaultdict(list)
    temperature = 1.0
    len_predictions = dur
    key_order = ['pitch', 'step', 'duration']

    cur_melody = mc.get_melody_data(filename)

    k = remove_duplicates([[a, float(b)]for a, b in cur_melody])
    t = merge_data(add_time_intervals(k))

    for key in t:
        test[key] += t[key]
    test['pitch'] = [librosa.note_to_midi(i) for i in test['pitch']]
    test = pd.DataFrame({name: np.array(value) for name, value in test.items()})

    sample_notes = np.stack([test[key] for key in key_order], axis=1)

    input_notes = (sample_notes[:seq_length] / np.array([vocab_size, 1, 1]))

    generated_notes = []
    prev_start = 0
    end = 0

    while end < len_predictions:
        pitch, step, duration = predict_next_note(input_notes, model_melody, temperature)
        start = max(prev_start + step, end)
        end = start + duration
        input_note = (pitch, step, duration)
        generated_notes.append((*input_note, start, end))
        input_notes = np.delete(input_notes, 0, axis=0)
        input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
        prev_start = start

    generated_notes = pd.DataFrame(generated_notes, columns=(*key_order, 'start', 'end'))

    return add_N(generated_notes)
    
