import numpy as np

def convolute3DVecMatrix(D, s):
    dim = D.shape
    result = np.zeros((dim[0], dim[1], dim[2], s.shape[3]))
    for n in range(dim[0]):
        for m in range(dim[1]):
            for l in range(dim[2]):
                for i in range(dim[0]):
                    for j in range(dim[1]):
                        for k in range(dim[2]):
                            temp = np.matmul(D[ (n-i) % dim[0], (m-j) % dim[1], (l-k) % dim[2]] , s[i,j,k])
                            result[n,m,l] += temp
    return result

#we dont use it, its just here to double check
def four3d(x, dim , dir = 1):
    result = np.array([1.j for i in range(dim[0]*dim[1]* dim[2])])
    result = result.reshape(dim)
    for i in range(dim[0]):
        for j in range(dim[1]):
            for k in range(dim[2]):
                for ip in range(dim[0]):
                    for jp in range(dim[1]):
                        for kp in range(dim[2]):
                            result[i,j,k] += x[ip,jp,kp] * np.exp(dir * -2 * np.pi * 1.j * ip * i / dim[0]) * np.exp(dir* - 2 * 1.j * np.pi * jp * j / dim[1]) * np.exp(dir * - 2* 1.j * np.pi * kp * k / dim[2])
    return result if dir>=0 else result/(dim[0]*dim[1]*dim[2])

def convolution1D(x,y):
    result = np.zeros(len(x))
    for i in range(len(x)):
        for j in range(len(x)):
            result[i] += x[j] * y[(i-j) % len(y)]
    return result

def four1D(x, dir = 1):
    N = len(x)
    result = np.array([0.j for i in range(N)])
    for i in range(N):
        for ip in range(N):
            result[i] += x[ip] * np.exp(-2 * dir * np.pi * 1.j * (ip*i)/N)
    return result if dir>=0 else result/N 