import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt




data, samplerate = sf.read('382769__martysonic__mighty-lava.wav')
sf.write('new_file.flac', data*200, samplerate)
sf.write('stereo_file.wav', np.random.randn(2000000, 2), 44100, 'PCM_24')
fig, ax = plt.subplots(nrows = 1, ncols = 2)
fig = plt.Figure()
ax[0].plot(data[:,0])
ax[1].plot(data[:,1])
plt.savefig("data_original")
# split data
data_split = data[data.shape[0]//2:,:]
fig, ax = plt.subplots(nrows = 1, ncols = 2)
fig = plt.Figure()
ax[0].plot(data_split[:,0])
ax[1].plot(data_split[:,1])
plt.savefig("data_split")
sf.write('new_file_splited.wav', data_split, samplerate)

print(data.shape)
'''
shape of data sound is (1411200, 2), with 2 columns is 2 chanels of sound,
the first column is the sound for the left channel (left speaker), the second column is the sound for the right channel (right speaker) 
'''

temp = data
# temp[:,0]=0
temp[:,1]=0
fig, ax = plt.subplots(nrows = 1, ncols = 2)
fig = plt.Figure()
ax[0].plot(temp[:,0])
ax[1].plot(temp[:,1])
plt.show()
sf.write('change2.wav', temp, samplerate)
