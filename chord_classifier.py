from chord_extractor.extractors import Chordino
import mingus.core.chords as chords


def transform_song_to_data(filename):

    chordino = Chordino()
    chords = chordino.extract(filename)
    data = []
    prev_len = 0

    for chord in chords:
        data.append((chord[0], chord[1]-prev_len))
        prev_len = chord[1]

    return data  # contains elements in format (chord, duration)
