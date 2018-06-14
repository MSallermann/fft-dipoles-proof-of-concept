import numpy as np
import util

#specify a lattice structure
bv = np.array(  [[1,0,0],
                 [0,1,0],
                 [0,0,1]]
             )
basis = np.array([[0,0,0]])
N = [3,1,1]

#specify outputfile
outputfile = "proof_output.txt"


#Set up a system
pos = util.setUpLattice(bv, N, basis)
spins = util.buildSpins(pos)

#Brute Force the gradients
gradBF = util.calculateGradientsBF(pos, spins)
print("Brute Force")
print(gradBF)
#Fourier Transformed grads
FgradBF = np.fft.fftn(gradBF)

#------------------------------
#Calculate Gradients with FFT
#------------------------------
#Build mtilde
mt = []
for b in basis:
    Nmt = 2 * N[0] * N[1] * N[2]
    mtemp = np.zeros(3*Nmt).reshape((Nmt, 3))
    for c in range(N[2]):
        for b in range(N[1]):
            for a in range(N[0]):
                mtemp[a + N[0]*b + N[0]*N[1]*c] = spins[a + N[0]*b + N[0]*N[1]*c]
    
    mt.append(mtemp)
mt = np.array(mt)

# build DMatrices
Dt=[]
for n in range(0,2*N[0]):
    r_diff = [0,0,0]
    if(n<N[0]):
        r_diff = n*bv[0]
    else:
        r_diff = (n-2*N[0])*bv[0]
    Dt.append(util.dipoleMatrix(r_diff))
Dt = np.array(Dt)

#Calculate the convolution
fmt=np.fft.fft(mt)
fDt=np.fft.fft(Dt)

print("With Convolutions")
res=[]
for i in range(N[0]):
    res.append(np.matmul(fDt[i], fmt[0,i]))
res = np.array(res)

print(res)
#-----------------------------------
# Write some results to file
#-----------------------------------

with open(outputfile,"w") as f:
    f.write("-------------------------------------\n")
    f.write("Bravais Lattice: \n" + str(bv) + "\n")
    f.write("N = " + str(N) + "\n")
    f.write("Basis: \n" + str(basis) + "\n")
    f.write("-------------------------------------\n")
    f.write("Original magnetization: \n" + str(spins) + "\n")
    f.write("Padded magnetization: \n" + str(mt)+ "\n")
    f.write("Fourier Transformed: \n" + str(fmt)+ "\n")
    f.write("Padded D-Matrices \n" + str(Dt)+ "\n")
    f.write("-------------------------------------\n")
    f.write("Brute Force Gradients: \n"+ str(gradBF) + "\n")
    f.write("Fourier Tansformed BF Gradients: \n" + str(FgradBF)+ "\n")
    f.write("Result of ptwise multiplication of Dt and mt: \n")
    f.write(str(res))



