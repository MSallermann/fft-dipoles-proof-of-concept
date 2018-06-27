import numpy as np
import import_python_util
from spirit_python_utilities import util, mathematics, graphing

np.random.seed(1337)
#specify outputfile
outputfile = "Output/output_proof_No_Basis.txt"

#specify a lattice structure
bv = np.array(  [[1,0,0],
                 [0,1,0],
                 [0,0,1]]
             )
basis = np.array([[0,0,0]])
N = [2,2,2]

#Set up a system
pos = util.setUpLattice(bv, N, basis)
spins = util.buildSpins(pos, "Random")

#Brute Force the gradients
gradBF = util.calculateGradientsBF(pos, spins)

#Fourier Transformed grads
FgradBF = np.fft.fftn(gradBF)

#Calculate the total number of "spins" after padding
it_c = N[2] if N[2]==1 else 2*N[2]-1
it_b = N[1] if N[1]==1 else 2*N[1]-1
it_a = N[0] if N[0]==1 else 2*N[0]-1
Npadding = it_a * it_b * it_c

#build padded magnetization mtilde
mt = np.zeros(3*Npadding).reshape((Npadding, 3))
for c in range(N[2]):
    for b in range(N[1]):
        for a in range(N[0]):
            mt[a + it_a*b + it_a*it_b*c] = spins[a + N[0]*b + N[0]*N[1]*c]
mt = np.array(mt)

# build DMatrices
Dt = np.ones(9*Npadding).reshape((Npadding,3,3))
for c in range(it_c):
    for b in range(it_b):
        for a in range(it_a):
            c_idx = -c if c<N[2] else 2 * N[2] - 1 - c
            b_idx = -b if b<N[1] else 2 * N[1] - 1 - b
            a_idx = -a if a<N[0] else 2 * N[0] - 1 - a 
            #print("adf",a_idx,b_idx,c_idx)
            Dt[a + it_a * b + it_b * it_c *c] = util.dipoleMatrix(a_idx*bv[0] + b_idx*bv[1] + c_idx*bv[2])

#Calculate the convolution directly
mt_np = util.convertToNumpyStyle(mt, [it_a, it_b, it_c])
Dt_np = util.convertToNumpyStyle(Dt, [it_a, it_b, it_c])
conv = mathematics.convolute3DVecMatrix(Dt_np, mt_np)
conv = util.convertToSpiritStyle(conv)


#Calculate the multi-dim FT with numpy
fmt = np.fft.fftn(mt_np, axes=[0,1,2])
fDt = np.fft.fftn(Dt_np, axes=[0,1,2])

#ptwise multiplication
res=np.array([0.j for i in range(it_a*it_b*it_c*3)]).reshape(it_a, it_b, it_c, 3)

for c in range(it_c):
    for b in range(it_b):
        for a in range(it_a):
            res[a,b,c] = np.matmul(fDt[a,b,c],fmt[a,b,c])


#Reverse FT
result = np.fft.ifftn(res, axes=[0,1,2])
res_final = -1 * util.convertToSpiritStyle(result)

#-----------------------------------
# Write some results to file
#-----------------------------------

with open(outputfile,"w") as f:
    np.set_printoptions(precision=3)
    f.write( "#-------------------------------------\n")
    f.write( "#            Geometry                 \n")
    f.write( "#-------------------------------------\n")
    f.write("Bravais Lattice: \n" + str(bv) + "\n")
    f.write("N = " + str(N) + "\n")
    f.write("Basis: \n" + str(basis) + "\n")
    f.write( "\n#-------------------------------------\n")
    f.write(   "#       Quantities           \n")
    f.write(   "#-------------------------------------\n")
    f.write("Original magnetization: \n" + str(spins) + "\n")
    f.write("Padded magnetization: \n" + str(mt)+ "\n")
    f.write("Fourier Transformed padded magnetization: \n" + str(fmt)+ "\n")
    f.write("Padded D-Matrices \n" + str(Dt)+ "\n")
    f.write("Fourier transformed padded D-Matrices \n" + str(fDt) + "\n")
    f.write("\n#-------------------------------------\n")
    f.write(  "#            Direct Conv.             \n")
    f.write(  "#-------------------------------------\n")
    f.write("Convolution: \n"+ str(conv) + "\n")
    f.write( "\n#-------------------------------------\n")
    f.write(   "#               Results               \n")
    f.write(   "#-------------------------------------\n")
    f.write("Brute Force Gradients: \n"+ str(gradBF) + "\n")
    f.write("With convolution Theorem: \n")
    f.write(str(res_final))



