from mingus.core import chords
import librosa


def decompose_chord(chord):
    if chord != 'N':  # 'N' means no chord is played
        return chords.from_shorthand(chord)  # transforming string chord into array of string notes
    else:
        return []
     
    from midiutil import MIDIFile


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
        # beat_duration = duration
        for note in decomposed_chord:
            midi_file.addNote(track, channel, librosa.note_to_midi(note + '3'), time, beat_duration, volume)
        time += beat_duration

    with open("result.mid", "wb") as output_file:
        midi_file.writeFile(output_file)
