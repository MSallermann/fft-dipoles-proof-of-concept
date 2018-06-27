import numpy as np


a = np.arange(9).reshape(3,3)
fft_a = np.fft.fftn(a)

print(a)
print(fft_a)

for i in range(3):
    print(np.fft.fft(a[i,:]))

