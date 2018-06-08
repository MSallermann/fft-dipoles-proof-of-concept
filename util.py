import numpy as np

def setUpLattice(bv, N, basis = [[0, 0, 0]]):
    result = []
    for c in range(N[2]):
        for b in range(N[1]):
            for a in range(N[0]):
                for base in basis:
                    result.append(bv[0] * a + bv[1] * b + bv[2] * c + base)
    return np.array(result)

def buildSpins(lattice, config = "Random"):
    if config == "Random":
        phi = np.random.rand(len(lattice))
        theta = np.random.rand(len(lattice))
        x = np.cos(phi*2*np.pi)*np.sin(theta*2*np.pi)
        y = np.sin(phi*2*np.pi)*np.sin(theta*2*np.pi)
        z = np.cos(theta*2*np.pi)
        result = np.dstack((x,y,z))
    return result

def dipoleMatrix(r_vect):
    x = r_vect[0]
    y = r_vect[1]
    z = r_vect[2]
    r = np.sqrt(r_vect[0]**2 + r_vect[1]**2 + r_vect[2]**2)

    result = 1/r**5*np.array(
                                [[x**2 - r**2, x*y, x*z],
                                [x*y, y**2-r**2, y*z],
                                [x*z, y*z, z**2-r**2]]
                            )
    return result



