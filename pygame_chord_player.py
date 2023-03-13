from librosa import note_to_midi
import pygame.midi
import time
import mingus.core.chords as chords


def decompose_chord(chord):
    if chord != 'N':  # 'N' means no chord is played
        return chords.from_shorthand(chord)  # transforming string chord into array of string notes
    else:
        return []


def play_chords(chord_data):
    pygame.init()
    pygame.midi.init()
    player = pygame.midi.Output(0)

    for chord, duration in chord_data:

        decomposed_chord = decompose_chord(chord)

        for note in decomposed_chord:
            oct_note = note + '3'
            midi_note = note_to_midi(oct_note)
            player.note_on(midi_note, 127)

        time.sleep(duration)

        for note in decomposed_chord:
            midi_note = note_to_midi(note)
            player.note_off(midi_note, 127)

    del player
    pygame.midi.quit()
