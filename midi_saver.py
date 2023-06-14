from midiutil import MIDIFile
from mingus.core import chords
import librosa


def decompose_chord(chord):
    if chord != 'N':  # 'N' means no chord is played
        return chords.from_shorthand(chord)  # transforming string chord into array of string notes
    else:
        return []


def save_chords_as_midi(chord_data, BPM=60, transpose=0):
    track = 0
    channel = 0
    time = 0  # In beats
    volume = 100  # 0-127, as per the MIDI standard
    midi_file = MIDIFile(4)
    midi_file.addTempo(track, time, BPM)

    for chord, duration in chord_data:
        decomposed_chord = decompose_chord(chord)
        beat_duration = duration * (BPM / 60)
        # beat_duration = duration
        for note in decomposed_chord:
            midi_file.addNote(track, channel, librosa.note_to_midi(note + '3')+transpose, time, beat_duration, volume)
        time += beat_duration

    with open("result.mid", "wb") as output_file:
        midi_file.writeFile(output_file)


def save_song_as_midi(chord_data, melody_data, BPM=60, transpose=0, melody_volume=100, chords_volume=100):

    chords_track = 0
    channel = 0
    time = 0  # In beats
    midi_file = MIDIFile(4)
    midi_file.addTempo(chords_track, time, BPM)

    for chord, duration in chord_data:
        decomposed_chord = decompose_chord(chord)
        beat_duration = duration * (BPM / 60)
        # beat_duration = duration
        for note in decomposed_chord:
            midi_file.addNote(chords_track, channel, librosa.note_to_midi(note + '3')+transpose, time, beat_duration, chords_volume)
        time += beat_duration

    melody_track = 1
    channel = 0
    time = 0  # In beats
    midi_file.addTempo(melody_track, time, BPM)

    for note, duration in melody_data:
        beat_duration = duration * (BPM / 60)
        if note != 'N':
            midi_file.addNote(melody_track, channel, librosa.note_to_midi(note)+transpose, time, beat_duration, melody_volume)
        time += beat_duration

    with open("result.mid", "wb") as output_file:
        midi_file.writeFile(output_file)
        
