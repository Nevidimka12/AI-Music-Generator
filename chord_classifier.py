from chord_extractor.extractors import Chordino

def detect_BPM(song_name):
    y, sr = librosa.load(song_name)
    y_percussive = librosa.effects.percussive(y)
    tempo, _ = librosa.beat.beat_track(y=y_percussive, sr=sr)
    return tempo

def transform_song_to_data(filename):

    chordino = Chordino()
    chords = chordino.extract(filename)
    data = []
    prev_len = 0

    for chord in chords:
        data.append((chord[0], chord[1]-prev_len))
        prev_len = chord[1]

    return data, detect_BPM(filename)  # data contains elements in format (chord, duration)
