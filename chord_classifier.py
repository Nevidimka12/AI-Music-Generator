import librosa
import vamp

def detect_BPM(song_name):
    y, sr = librosa.load(song_name)
    y_percussive = librosa.effects.percussive(y)
    tempo, _ = librosa.beat.beat_track(y=y_percussive, sr=sr)
    return tempo


def transform_song_to_data(filename):

    y, sr = librosa.load(filename)

    chords = vamp.collect(y, sr, 'nnls-chroma:chordino')

    data = []
    prev_len = 0

    for i in range(len(chords['list'])):
        element = chords['list'][i]
        label = str(element['label'])
        timestamp = float(element['timestamp'])
        #print(type(chords[lis]))
        data.append((label, timestamp - prev_len))
        prev_len = timestamp

    return data # data contains elements in format (chord, duration)
