import numpy as np
import util

#specify a lattice structure
bv = np.array(  [[1,0,0],
                [0,1,0],
                [0,0,1]]
             )
basis = np.array([[0,0,0], [0.3,0.6,0]])
N = [3,3,1]

#Set up a system
pos = util.setUpLattice(bv, N, basis)
spins = util.buildSpins(pos)

#Brute Force the gradients
gradBF = util.calculateGradientsBF(pos, spins)

#Calculate Gradients with FFT

#Build mtilde
mt = []
for b in basis:
    Nmt = 8 * N[0] * N[1] * N[2]
    mtemp = np.zeros(3*Nmt).reshape((Nmt, 3))
    
    for c in range(N[2]):
        for b in range(N[1]):
            for a in range(N[0]):
                mtemp[a + N[0]*b + N[0]*N[1]*c] = spins[a + N[0]*b + N[0]*N[1]*c]
    
    mt.append(mtemp)
mt = np.array(mt)

#build DMatrices
Dt=[]
for b in range(len(basis)):
    for bp in range(b,len(basis)):
        Dtemp=np.zeros()

