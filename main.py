import pygame_chord_player
import chord_classifier

data = chord_classifier.transform_song_to_data('song1.wav')
pygame_chord_player.play_chords(data)
