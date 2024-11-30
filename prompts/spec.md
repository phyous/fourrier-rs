Build an applicaiton that takes as input a audio file (wav, mp3, etc) and outputs a visual representation of the audio to the console.

We should process the file and:
1/ Run speech recognition (using openai's whisper model - that we can run locally). We want to get words with timestamps
2/ Extract the waveform of the audio over time 
3/ Extract the spectrogram of the audio over time

We'll then render in the console the 1/ words with time boundaries, 2/ the waveform, and 3/ the spectrogram – all in the console using an advanced terminal rendering library (colors, etc).
