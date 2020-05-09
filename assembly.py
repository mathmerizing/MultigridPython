import numpy as np
from sympy import symbols, diff, integrate
from scipy.sparse import dok_matrix

def getLocalMatrices(degree = 1):
    """
    Assemble the local stiffness matrices K = [K_xx, K_xy, K_yy]
    and the local mass matrix M.
    """
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
        phi = [
            (1-x-y)*(1-2*x-2*y),
            x*(2*x-1),
            y*(2*y-1),
            4*x*(1-x-y),
            4*x*y,
            4*y*(1-x-y)
        ]
    elif degree == 3:
        # cubic basis
        phi = [
            (1-x-y)*(1-3*x-3*y)*(2-3*x-3*y),
            x*(3*x-1)*(3*x-2),
            y*(3*y-1)*(3*y-2),
            9/2*x*(1-x-y)*(2-3*x-3*y),
            9/2*x*(1-x-y)*(3*x+3*y-1),
            9/2*x*y*(3*x-2),
            9/2*x*y*(3*y-2),
            9/2*y*(1-x-y)*(3*x+3*y-1),
            9/2*y*(1-x-y)*(2-3*x-3*y),
            27*x*y*(1-x-y)
        ]

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

def assembleSystem(grid, K, M):
    """
    Assemble system matrix and right hand side with the help of the local matrices
    and the information stored in the grid.
    """
    numDofs = grid.getActiveNodes()
    systemMatrix    = dok_matrix((numDofs,numDofs),dtype=np.float32)
    systemRightHand = np.zeros((numDofs,1),dtype=np.float32)

    for triangle in grid.triangles:
        for firtsNode in triangle.nodes:
            for secondNode in triangle.nodes:
                # TODO
                pass

    return systemMatrix, systemRightHand