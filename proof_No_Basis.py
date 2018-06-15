import numpy as np
import import_python_util
from spirit_python_utilities import util, math, graphing

np.random.seed(1337)
#specify outputfile
outputfile = "Output/output_proof_No_Basis.txt"

#specify a lattice structure
bv = np.array(  [[1,0,0],
                 [0,1,0],
                 [0,0,1]]
             )
basis = np.array([[0,0,0]])
N = [3,1,1]

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

#Calculate the total number of "spins" after padding
Npadding = 1
if N[0]>1:
    Npadding *= (2*N[0]-2)
if N[1]>1:
    Npadding *= (2*N[1]-2)
if N[2]>1:
    Npadding *= (2*N[2]-2)
mt = np.zeros(3*Npadding).reshape((Npadding, 3))

for c in range(N[2]):
    for b in range(N[1]):
        for a in range(N[0]):
            mt[a + N[0]*b + N[0]*N[1]*c] = spins[a + N[0]*b + N[0]*N[1]*c]

mt = np.array(mt)

# build DMatrices
Dt = np.zeros(9*Npadding).reshape((Npadding,3,3))
iterate_c = N[2] if N[2]==1 else 2*N[2]-2
iterate_b = N[1] if N[1]==1 else 2*N[1]-2
iterate_a = N[0] if N[0]==1 else 2*N[0]-2

for c in range(iterate_c):
    for b in range(iterate_b):
        for a in range(iterate_a):
            c_idx = c if c<N[2] else 2*N[2]-2-c
            b_idx = b if b<N[1] else 2*N[1]-2-b
            a_idx = a if a<N[0] else 2*N[0]-2-a
            Dt[a + N[0]*b + N[0]*N[1]*c] = util.dipoleMatrix(a_idx*bv[0] + b_idx*bv[1] + c_idx*bv[2])

#Calculate the convolution
#Calculate the mulitdim FT with numpy

fmt=np.fft.fftn(mt.reshape((iterate_c, iterate_b, iterate_a,3)), axes=[0,1,2]).reshape(Npadding,3)
fDt=np.fft.fftn(Dt.reshape((iterate_c, iterate_b, iterate_a,3,3)), axes=[0,1,2]).reshape((Npadding,3,3))

print(fDt.shape)
print(fmt.shape)

res=[]
for i in range(Npadding):
    res.append(np.matmul(fDt[i], fmt[i]))
res = np.array(res)

res=res.reshape([iterate_c, iterate_b, iterate_a,3])
result = np.fft.ifftn(res).reshape(Npadding,3)


#Reverse FT
print(res)


#-----------------------------------
# Write some results to file
#-----------------------------------

with open(outputfile,"w") as f:
    np.set_printoptions(precision=3)
    f.write("-------------------------------------\n")
    f.write("Bravais Lattice: \n" + str(bv) + "\n")
    f.write("N = " + str(N) + "\n")
    f.write("Basis: \n" + str(basis) + "\n")
    f.write("-------------------------------------\n")
    f.write("Original magnetization: \n" + str(spins) + "\n")
    f.write("Padded magnetization: \n" + str(mt)+ "\n")
    f.write("Fourier Transformed padded magnetization: \n" + str(fmt)+ "\n")
    f.write("Padded D-Matrices \n" + str(Dt)+ "\n")
    f.write("Fourier transformed padded D-Matrices \n" + str(fDt) + "\n")
    f.write("-------------------------------------\n")
    f.write("Brute Force Gradients: \n"+ str(gradBF) + "\n")
    f.write("Fourier Tansformed BF Gradients: \n" + str(FgradBF)+ "\n")
    f.write("Result of ptwise multiplication of Dt and mt: \n")
    f.write(str(result))



