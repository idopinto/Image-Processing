# Fourier Transform & Convolution
## Exercise 2 | Image Processing @ HUJI

### Goals
Understanding the concept of the frequency domain by performing
simple manipulations on sounds and images. 

### This exercise covers
• Implementing Discrete Fourier Transform (DFT) on 1D and 2D signals

• Performing sound fast forward

• Performing image derivative


### Original wav file.
[aria_4kHz.wav](external%2Faria_4kHz.wav)

### X2 Fast Forward by changing sample rate
[change_rate.wav](results%2Faudio%2Fchange_rate.wav)
### X2 Fast Forward using Fourier Transform
[change_samples.wav](results%2Faudio%2Fchange_samples.wav)
### ## X2 Fast Forward using spectrogram (stft on each row of the spectrogram)
[spectogram_only.wav](results%2Faudio%2Fspectogram_only.wav)

### X2 Fast Forward using spectrogram and phase vocoder
[with_vocoder.wav](results%2Faudio%2Fwith_vocoder.wav)

## Image derivatives
![original_monkey.png](results%2Fimages%2Foriginal_monkey.png)

### Image derivatives in image space
![monkey_conv_der.png](results%2Fimages%2Fmonkey_conv_der.png)
###  Image derivatives in fourier space
![monkey_fourier_der.png](results%2Fimages%2Fmonkey_fourier_der.png)