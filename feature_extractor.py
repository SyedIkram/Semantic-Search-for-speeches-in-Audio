import os
import numpy as np
import soundfile
from numpy.lib.stride_tricks import as_strided


class FeatureExtractor():

    def __init__(self,step=10, window=20, eps=1e-14):
        self.feature_dim = int(0.001 * window * 16000) + 1
        self.window = window
        self.step = step
        self.eps = eps

    def spectrogram(self,filename, step=10, window=20, max_freq=8000, eps=1e-14):

        with soundfile.SoundFile(filename) as sound_file:
            audio = sound_file.read(dtype='float32')
            sample_rate = sound_file.samplerate
            if audio.ndim >= 2:
                audio = np.mean(audio, 1)
            
            hop_length = int(0.001 * step * sample_rate)
            fft_length = int(0.001 * window * sample_rate)

            pxx, freqs = self.calc_spectrogram(
                audio, fft_length=fft_length, sample_rate=sample_rate,
                hop_length=hop_length)
            ind = np.where(freqs <= max_freq)[0][-1] + 1
        return np.transpose(np.log(pxx[:ind, :] + eps))

    def calc_spectrogram(self, samples, fft_length, sample_rate, hop_length):

        window = np.hanning(fft_length)[:, None]
        window_norm = np.sum(window**2)

        scale = window_norm * sample_rate

        trunc = (len(samples) - fft_length) % hop_length
        x = samples[:len(samples) - trunc]

        nshape = (fft_length, (len(x) - fft_length) // hop_length + 1)
        nstrides = (x.strides[0], x.strides[0] * hop_length)
        x = as_strided(x, shape=nshape, strides=nstrides)

        assert np.all(x[:, 1] == samples[hop_length:(hop_length + fft_length)])

        x = np.fft.rfft(x * window, axis=0)
        x = np.absolute(x)**2

        x[1:-1, :] *= (2.0 / scale)
        x[(0, -1), :] /= scale

        freqs = float(sample_rate) / fft_length * np.arange(x.shape[0])

        return x, freqs

    def __call__(self, filename):
        return self.spectrogram(filename)