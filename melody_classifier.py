import numpy
import essentia.standard as es
import librosa
from chord_classifier import detect_BPM

from midiutil import MIDIFile


def transform_to_melody_data(pitch_values, pitch_times):
    return [(pitch_values[i], pitch_times[i + 1] - pitch_times[i]) for i in range(len(pitch_values) - 1)]


def remove_duplicates(melody_data_hz, min_freq, min_note_length):
    melody_data = list(map(lambda x: (librosa.hz_to_note(x[0]), x[1]) if x[0] >= min_freq else ('N', x[1]), melody_data_hz))
    res = []
    duplicates_duration = 0
    for i in range(len(melody_data) - 1):

        duplicates_duration += melody_data[i][1]

        if melody_data[i][0] != melody_data[i + 1][0]:
            if duplicates_duration >= min_note_length:
                res.append((melody_data[i][0], duplicates_duration))
            else:
                res.append(('N', duplicates_duration))
            duplicates_duration = 0

    if duplicates_duration == 0:
        res.append(melody_data[-1])
    else:
        duplicates_duration += melody_data[-1][1]
        res.append((melody_data[-1][0], duplicates_duration))

    res2 = []
    duplicates_duration = 0
    for i in range(len(res) - 1):

        duplicates_duration += res[i][1]

        if res[i][0] != res[i + 1][0]:

            res2.append((res[i][0], duplicates_duration))
            duplicates_duration = 0

    return res2


def save_notes_as_midi(note_data, BPM):
    track = 0
    channel = 0
    time = 0  # In beats
    volume = 100  # 0-127, as per the MIDI standard
    midi_file = MIDIFile(4)
    midi_file.addTempo(track, time, BPM)

    for note, duration in note_data:

        beat_duration = duration * (BPM / 60)
        # beat_duration = duration

        if note != 'N':
            midi_file.addNote(track, channel, librosa.note_to_midi(note), time, beat_duration, volume)

        time += beat_duration

    with open("result.mid", "wb") as output_file:
        midi_file.writeFile(output_file)


def get_melody_data(audiofile):

    loader = es.EqloudLoader(filename=audiofile, sampleRate=44100)
    audio = loader()

    pitch_extractor = es.PredominantPitchMelodia(frameSize=2048, hopSize=128, guessUnvoiced=True, filterIterations=15, minDuration=10)
    pitch_values, _ = pitch_extractor(audio)

    pitch_times = numpy.linspace(0.0, len(audio) / 44100.0, len(pitch_values))

    melody_data = transform_to_melody_data(pitch_values, pitch_times)

    min_note_len = (60/detect_BPM(audiofile)) * 0.125

    melody_data = remove_duplicates(melody_data, min_freq=20, min_note_length=min_note_len)  # Обработали melody_data, на этом можно обучаться как на chord_data

    return melody_data

    # save_notes_as_midi(melody_data)  # Если хочется послушать, что предсказали
