import librosa
import numpy as np
import soundfile


def get_key(waveform, sr, tstart=None, tend=None):

    if tstart is not None:
        tstart = librosa.time_to_samples(tstart, sr=sr)
    if tend is not None:
        tend = librosa.time_to_samples(tend, sr=sr)

    y_segment = waveform[tstart:tend]
    chromograph = librosa.feature.chroma_cqt(y=y_segment, sr=sr, bins_per_octave=24)

    chroma_vals = []
    for i in range(12):
        chroma_vals.append(np.sum(chromograph[i]))
    pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    keyfreqs = {pitches[i]: chroma_vals[i] for i in range(12)}

    keys = [pitches[i] for i in range(12)] + [pitches[i] + 'm' for i in range(12)]

    maj_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    min_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

    min_key_corrs = []
    maj_key_corrs = []
    for i in range(12):
        key_test = [keyfreqs.get(pitches[(i + m) % 12]) for m in range(12)]
        maj_key_corrs.append(round(np.corrcoef(maj_profile, key_test)[1, 0], 3))
        min_key_corrs.append(round(np.corrcoef(min_profile, key_test)[1, 0], 3))

    key_dict = {**{keys[i]: maj_key_corrs[i] for i in range(12)},
                     **{keys[i + 12]: min_key_corrs[i] for i in range(12)}}

    key = max(key_dict, key=key_dict.get)
    bestcorr = max(key_dict.values())

    return key, bestcorr


def key_diff(key) -> int:    # How much should you transpose given key to get C major / A minor
    if 'm' not in key:     # If major key (no m symbol)
        scale = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')
    else:
        scale = ('Am', 'A#m', 'Bm', 'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m')

    return -scale.index(key) if (scale.index(key) < 12 - scale.index(key)) else (12 - scale.index(key))


def transpose_song(input_path, output_path, threshold):

    y, sr = librosa.load(input_path)
    key, accuracy = get_key(y, sr)

    if accuracy < threshold:
        print("Threshold not passed!")
        return -1

    y_transposed = librosa.effects.pitch_shift(y, sr=sr, n_steps=key_diff(key))

    soundfile.write(output_path, y_transposed, samplerate=sr)


song_directory = "path to your song"
result_directory = "output_filename (also include format like mp3 or wav)"

transpose_song(song_directory, result_directory, threshold=0.9)
