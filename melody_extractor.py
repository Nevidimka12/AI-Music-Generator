import numpy
import essentia.standard as es
import librosa

from midiutil import MIDIFile


def transform_to_melody_data(pitch_values, pitch_times):
  return [(pitch_values[i], pitch_times[i+1]-pitch_times[i]) for i in range(len(pitch_values)-1)]


def remove_duplicates(a, min_freq=0.1, min_note_length=0.05):

  ans = []
  dur = 0
  for i in range(len(a)-1):
    dur += a[i][1]

    current_note = librosa.hz_to_note(a[i][0]) if a[i][0] >= min_freq else 'N'
    next_note = librosa.hz_to_note(a[i+1][0]) if a[i+1][0] >= min_freq else 'N'

    if current_note != next_note:
      ans.append((current_note, dur))
      dur = 0
      
  if a[-1][0] >= min_freq:
    ans.append((librosa.hz_to_note(a[-1][0]), a[-1][1]+dur))

  ans = [e if e[1] >= min_note_length else ('N', e[1]) for e in ans]

  return ans


def save_notes_as_midi(note_data, BPM=60):

    track = 0
    channel = 0
    time = 0  # In beats
    volume = 100  # 0-127, as per the MIDI standard
    midi_file = MIDIFile(4)
    midi_file.addTempo(track, time, BPM)

    for note, duration in note_data:
        
        beat_duration = duration * (BPM/60)
        # beat_duration = duration

        if note != 'N':
          midi_file.addNote(track, channel, librosa.note_to_midi(note), time, beat_duration, volume)
          
        time += beat_duration

    with open("result.mid", "wb") as output_file:
        midi_file.writeFile(output_file)



audiofile = 'path_to_audio'

loader = es.EqloudLoader(filename=audiofile, sampleRate=44100)
audio = loader()

pitch_extractor = es.PredominantPitchMelodia(frameSize = 2048, hopSize = 1024, guessUnvoiced = True)
pitch_values, _ = pitch_extractor(audio)

pitch_times = numpy.linspace(0.0,len(audio)/44100.0,len(pitch_values))

melody_data = transform_to_melody_data(pitch_values, pitch_times)
melody_data = remove_duplicates(melody_data, min_freq=20, min_note_length=0.12) # Обработали melody_data, на этом можно обучаться как на chord_data

save_notes_as_midi(melody_data) # Если хочется послушать, что предсказали
