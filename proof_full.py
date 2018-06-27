import numpy as np
import import_python_util
from spirit_python_utilities import util, mathematics, graphing
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------
#   BRIEF DESCRIPTION
#
#   Proof-of-concept implementation of an FFT-algorithm
#   to calculate gradients due to magnetic dipole-dipole interactions
#   on a bravais lattice with basis and open boundary conditions
#   
#   DISCLAIMER: This is not intended to be a reference implementation and 
#               it is likely to be very slow.
#               The only purpose of this code is to show that the basic
#               math and logic of the algorithm works correctly.
#
#   Let B be the number of atoms per basis cell
#
#   Steps:
#   1.  Separate the magnetic moments in B sets so that each sets only 
#       contains magnetic moments that live on the same sublattice
#   2.  Calculate equally as many sets of dipole-matrices where each set
#       describes interactions between two specific sublattices
#   3.  Pad the sets of dipole matrices and magnetic moments, so that the
#       convolution of the padded sets is equal to the correct physical
#       calculation (with respect to the boundary condition)
#   4.  Perform the convolutions via the convolution theorem and add them
#------------------------------------------------------------------------


#------------------------------------------------------------------------
#                           SETUP
#------------------------------------------------------------------------

#Seed for RNG 
np.random.seed(1337)

#if plots should be made
make_plot = True

#specify outputfile
outputfile = "Output/output_full.txt"

#specify a lattice structure
#Please NOTICE: dont make the system too big as this implementation is really SLOW!
bv = np.array(  [[1,0,0],
                 [0.4,1,0],
                 [0,0,1]]
             )
basis = np.array([[0,0,0], [0.7,0.5,0]])
N = [5,5,5]
B = len(basis)

#Set up a system
pos = util.setUpLattice(bv, N, basis)
spins = util.buildSpins(pos, "Random")

#Brute Force the gradients
gradBF = util.calculateGradientsBF(pos, spins)


#------------------------------------------------------------------------
#                      CALCULATE PADDED QUANTITIES
#------------------------------------------------------------------------

#Calculate the number of "spins" after padding in each direction per sublattice
it_c = 2*N[2]-1
it_b = 2*N[1]-1
it_a = 2*N[0]-1
Npadding = it_a * it_b * it_c

#Build padded magnetizations
m_pad = []
for ib,b in enumerate(basis):
    mt = np.zeros(3*Npadding).reshape((Npadding, 3))
    for c in range(N[2]):
        for b in range(N[1]):
            for a in range(N[0]):
                mt[a + it_a*b + it_a*it_b*c] = spins[a*B + N[0]*B*b + N[0]*N[1]*B*c + ib]
    m_pad.append(mt)

#Build padded DMatrices
D_pad = np.zeros((B,B,Npadding,3,3))
for i1,b1 in enumerate(basis):
    for i2,b2 in enumerate(basis):
        for c in range(it_c):
            for b in range(it_b):
                for a in range(it_a):
                    c_idx = - c if c<N[2] else 2 * N[2] - 1 - c
                    b_idx = - b if b<N[1] else 2 * N[1] - 1 - b
                    a_idx = - a if a<N[0] else 2 * N[0] - 1 - a
                    D_pad[i1, i2, a + it_a * b + it_b * it_c *c] = util.dipoleMatrix(a_idx*bv[0] + b_idx*bv[1] + c_idx*bv[2] + b2 - b1)


#----------------------------------------------------------------------
# Calculate the convolutions directly (without conv. theorem)
#----------------------------------------------------------------------

conv_sublattices = []
for i in range(B):
    conv = np.zeros((it_a, it_b, it_c, 3))
    for j in range(B):
        Dt = D_pad[i,j]
        mt = m_pad[j]
        mt_np = util.convertToNumpyStyle(mt, [it_a, it_b, it_c])
        Dt_np = util.convertToNumpyStyle(Dt, [it_a, it_b, it_c])
        conv += mathematics.convolute3DVecMatrix(Dt_np, mt_np)
    conv = util.convertToSpiritStyle(conv)
    conv_sublattices.append(conv)
conv_sublattices = np.array(conv_sublattices)

print(conv_sublattices.shape)
conv = util.joinSublattices(conv_sublattices, [it_a, it_b, it_c])

#-----------------------------------------------------------------------
# Use the convolution theorem for the calculation
#-----------------------------------------------------------------------

conv_ft = []
for i in range(B):
    conv = np.array([0.j for i in range(it_a * it_b * it_c * 3)]).reshape(it_a, it_b, it_c, 3)
    for j in range(B):
        #Preparation
        Dt = D_pad[i,j]
        mt = m_pad[j]
        mt_np = util.convertToNumpyStyle(mt, [it_a, it_b, it_c])
        Dt_np = util.convertToNumpyStyle(Dt, [it_a, it_b, it_c])

        #Perform Fft with numpy
        fmt = np.fft.fftn(mt_np, axes=[0,1,2])
        fDt = np.fft.fftn(Dt_np, axes=[0,1,2])

        # elementwise multiplication
        res=np.array([0.j for i in range(it_a*it_b*it_c*3)]).reshape(it_a, it_b, it_c, 3)
        for c in range(it_c):
            for b in range(it_b):
                for a in range(it_a):
                    res[a,b,c] = np.matmul(fDt[a,b,c],fmt[a,b,c])
        #Reverse FT
        conv+=np.fft.ifftn(res, axes=[0,1,2])
    conv = util.convertToSpiritStyle(conv)
    conv_ft.append(conv)
conv_ft = np.array(conv_ft)
print(conv_ft.shape)
res_final = util.joinSublattices(conv_ft, [it_a, it_b, it_c])


#------------------------------------------------------------------------
# Write some results to file
#------------------------------------------------------------------------

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
    f.write("\nPadded magnetizations:")
    for i, mt in enumerate(m_pad):
        f.write("\nSublattice: "+str(i)+"\n")
        f.write(str(mt))
    f.write("\nPadded D-Matrices")
    for i in range(B):
        for j in range(B):
            f.write("\nSublattice: {0}, {1}\n".format(i,j))
            f.write(str(D_pad[i,j]))
    #f.write("Fourier Transformed padded magnetization: \n" + str(fmt)+ "\n")
   
    #f.write("Fourier transformed padded D-Matrices \n" + str(fDt) + "\n")
    
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
    
if make_plot:
    #Plot the original geometry
    fig, ax = plt.subplots(1)
    graphing.plotGeometry(pos, B, ax, show = False)
    fig.savefig("original_geometry")

    #Plot padded geometry


