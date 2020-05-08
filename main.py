import numpy as np
from sympy import *

GLOBAL_REFINEMENTS = 1;

def getLocalMatrices(degree = 1):
    x, y = symbols('x y')

    if degree == 1:
        # linear basis
        phi = [
            1-x-y,
            x,
            y
        ]
    elif degree == 2:
        # quadratic basis
        pass
    elif degree == 3:
        # cubic basis
        pass

    # compute derivatives of basis in x and y direction
    phi_x = [diff(phi_i, x) for phi_i in phi]
    phi_y = [diff(phi_i, y) for phi_i in phi]

    # compute local stiffness matrices
    K = []
    for (f,g) in [(phi_x,phi_x), (phi_x,phi_y), (phi_y,phi_y)]:
        n, m = len(f), len(g)
        mat  = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                mat[i,j] = integrate(
                            f[i] * g[j],
                            (x,0,1-y),
                            (y,0,1)
                )

        K.append(mat)

    # compute local mass matrix
    dim = len(phi)
    M   = np.zeros((dim, dim))

    for i in range(dim):
        for j in range(dim):
            M[i,j] = integrate(
                        phi[i] * phi[j],
                        (x,0,1-y),
                        (y,0,1)
            )

    return K, M



if __name__ == "__main__":
    # 1. getLocalMatrices
    K , M = getLocalMatrices(degree = 1)

    # 2. getCoarseGrid

    # 3. assmeble, apply BC

    # 4. global grid refinement + assembly finer grid matrices

    # 5. Multigrid ...
