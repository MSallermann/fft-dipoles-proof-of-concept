import numpy as np
import util

np.random.seed(1337)
outputfile = "Output/output_proof_1D"

#specify m and d
dim = 4

m = np.random.random(dim)
d = np.random.random(dim)

#Build mtilde
mt = np.array([m[i] if i<dim else 0 for i in range(2*dim-2)])
#Build dtilde
dt = np.array([d[i] if i<dim else d[2*dim-2-i] for i in range(2*dim-2)])
dt[0] = 0

#Calculate BF.
gradBF = np.zeros(dim)
for i in range(dim):
    for j in range(dim):
        if i==j:
            continue
        gradBF[i] += d[np.abs(i-j)] * m[j]

FgradBF = np.fft.fftn(gradBF)

#Using FT
fmt=np.fft.fftn(mt)
fdt=np.fft.fftn(dt)

#multiply
res = util.four1D( fdt * fmt , -1)

fmt = util.four1D(mt)
fdt = util.four1D(dt)

conv = util.convolution1D(dt, mt)
convnp = np.convolve(mt, dt, mode="same")


#-----------------------------------
# Write results to file
#-----------------------------------

with open(outputfile,"w") as f:
    np.set_printoptions(precision=3)
    f.write( "#-------------------------------------\n")
    f.write( "#       Original Quantities            \n")
    f.write( "#-------------------------------------\n")
    f.write("Original magnetization: \n" + str(m) + "\n")
    f.write("Original D-Matrices \n" + str(d)+ "\n")
    f.write("\n#-------------------------------------\n")
    f.write(  "#       Padded Quantities            \n")
    f.write(  "#-------------------------------------\n")
    f.write("Padded magnetization: \n" + str(mt)+ "\n")
    f.write("Padded D-Matrices \n" + str(dt)+ "\n")
    f.write("\n#-------------------------------------\n")
    f.write(  "#   Fourier Transformed Quantities    \n")
    f.write(  "#-------------------------------------\n")
    f.write("Fourier Transformed padded magnetization: \n" + str(fmt)+ "\n")
    f.write("Fourier Transformed padded d-matrices: \n" + str(fdt)+ "\n")
    f.write("\n#-------------------------------------\n")
    f.write(  "#            Results                  \n")
    f.write(  "#-------------------------------------\n")
    f.write("Brute Force Gradients: \n"+ str(gradBF) + "\n")
    f.write("Convolution of padded mt * dt (own) \n")
    f.write(str(conv)+"\n")
    #f.write("Convolution of padded mt * dt (numpy) \n")
    #f.write(str(convnp)+"\n")
    f.write("-------------------------------------\n")
    #f.write("Fourier Tansformed BF Gradients: \n" + str(FgradBF)+ "\n")
    f.write("Result of convolution Theorem: \n")
    f.write(str(res))

