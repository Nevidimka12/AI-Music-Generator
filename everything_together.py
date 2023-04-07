from chord_extractor.extractors import Chordino
import librosa
from midiutil import MIDIFile


def decompose_chord(chord):
    if chord != 'N':  # 'N' means no chord is played
        return chords.from_shorthand(chord)  # transforming string chord into array of string notes
    else:
        return []

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

    return (data, detect_BPM(filename))  # data contains elements in format (chord, duration)

def save_chords_as_midi(chord_data, BPM=60):

    track = 0
    channel = 0
    time = 0  # In beats
    volume = 100  # 0-127, as per the MIDI standard
    midi_file = MIDIFile(4)
    midi_file.addTempo(track, time, BPM)

    for chord, duration in chord_data:
        decomposed_chord = decompose_chord(chord)
        beat_duration = duration * (BPM/60)
        for note in decomposed_chord:
            midi_file.addNote(track, channel, librosa.note_to_midi(note + '3'), time, beat_duration, volume)
        time += beat_duration

    with open("result.mid", "wb") as output_file:
        midi_file.writeFile(output_file)


data, BPM = transform_song_to_data('song1.wav')
save_chords_as_midi(data, BPM)
