import numpy as np
import datetime

def convolute(x, y, dim):
    result = np.zeros((dim[0],dim[1],dim[2]))
    for n in range(dim[0]):
        for m in range(dim[1]):
            for l in range(dim[2]):
                for i in range(dim[0]):
                    for j in range(dim[1]):
                        for k in range(dim[2]):
                            result[n,m,l] += x[((n-i) % dim[0]), ((m-j) % dim[1]), ((l-k) % dim[2])] * y[i,j,k]
    return result

def FT3d(x, dim , dir = 1):
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



if __name__ == "__main___":
    dim = (3,3,3)
    x = np.arange(dim**3)
    x = np.random.random(dim**3)
    x = x.reshape((dim,dim,dim))
    y = np.arange(10,dim**3+10)
    y = np.random.random(dim**3)
    y = y.reshape((dim,dim,dim))
    cxy = convolute(x,y, dim)
    fcxy = FT3d(cxy, dim)
    fx = FT3d(x, dim)
    fx = np.fft.fftn(x)
    fy = FT3d(y, dim)
    fy = np.fft.fftn(y)
    f = open("out.txt",'w')
    f.write("Last changed: " + str(datetime.datetime.now()) + '\n') 
    f.write("Output\n")
    f.write('\nx\n')
    f.write(str(x))
    f.write('\ny\n')
    f.write(str(y))
    f.write('\nfx\n')
    f.write(str(fx))
    f.write('\nfy\n')
    f.write(str(fy))
    f.write('numpy fft x \n')
    f.write(str(np.fft.fftn(x)))
    f.write('\ncxy\n')
    f.write(str(cxy))
    f.write('\nfx*fy\n')
    f.write(str(fx*fy))
    f.write('\nfcxy\n')
    f.write(str(fcxy))
    f.close()