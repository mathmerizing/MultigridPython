import numpy as np
from sympy import symbols, diff, integrate
from scipy.sparse import csr_matrix
from grid import BoundaryCondition

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
    numDofs = len(grid.dofs)
    systemMatrix    = csr_matrix((numDofs,numDofs),dtype=np.float32)
    systemRightHand = np.zeros(numDofs,dtype=np.float32)

    for triangle in grid.triangles:
        # get all important triangle/ material parameters
        detJ, G = triangle.jacobi()
        # note: G = J^{-1} * J^{-T}
        a       = triangle.material.get("a")
        c       = triangle.material.get("c")
        f       = triangle.material.get("f")
        numLocalDofs = len(triangle.dofs)

        # assemble cell matrix and cell right hand side
        cellMatrix    = np.zeros((numLocalDofs, numLocalDofs), dtype=np.float32)
        cellRightHand = np.zeros(numLocalDofs, dtype=np.float32)

        # compute cellMatrix
        diffusionMatrix = a * (G[0,0]*K[0] + G[0,1]*K[1] + G[1,0]*K[1].T + G[1,1]*K[2])
        reactionMatrix  = c * M * detJ
        cellMatrix = diffusionMatrix + reactionMatrix

        # compute cellRightHand
        cellRightHand = f * detJ * np.sum(M, axis = 1)

        # write local to global matrix
        for i, firstDof in enumerate(triangle.dofs):
            systemRightHand[firstDof.ind] += cellRightHand[i]
            for j, secondDof in enumerate(triangle.dofs):
                systemMatrix[firstDof.ind, secondDof.ind] += cellMatrix[i,j]

    return systemMatrix, systemRightHand

def applyBoundaryCondition(grid,systemMatrix,systemRightHand):
    for edge in grid.edges:
        if(edge.boundaryConstraint is not None):
            if (edge.boundaryConstraint.type == "Dirichlet"):
                for dof in edge.dofs:
                    systemRightHand[dof.ind] = edge.boundaryConstraint.function(dof.x,dof.y)
                    # set row dof.ind to 0.0
                    systemMatrix.data[systemMatrix.indptr[dof.ind]:systemMatrix.indptr[dof.ind+1]] = 0.0
                    systemMatrix[dof.ind, dof.ind] = 1.0
            if (edge.boundaryConstraint.type == "Neumann"):
                pass
