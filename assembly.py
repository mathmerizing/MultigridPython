import numpy as np
from scipy.sparse import dok_matrix, csr_matrix
from grid import BoundaryCondition


def getLocalMatrices(degree=1):
    """
    Assemble the local stiffness matrices K = [K_xx, K_xy, K_yy]
    and the local mass matrix M.
    """
    assert degree in [1, 2], f"Degree {degree} is not currently supported"

    if degree == 1:
        K_xx = np.array(
            [[0.5, -0.5, 0.0], [-0.5, 0.5, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32
        )

        K_xy = np.array(
            [[0.5, 0.0, -0.5], [-0.5, 0.0, 0.5], [0.0, 0.0, 0.0]], dtype=np.float32
        )

        K_yy = np.array(
            [[0.5, 0.0, -0.5], [0.0, 0.0, 0.0], [-0.5, 0.0, 0.5]], dtype=np.float32
        )

        M = np.array(
            [
                [0.08333334, 0.04166667, 0.04166667],
                [0.04166667, 0.08333334, 0.04166667],
                [0.04166667, 0.04166667, 0.08333334],
            ],
            dtype=np.float32,
        )

        return [K_xx, K_xy, K_yy], M

    elif degree == 2:
        K_xx = np.array(
            [
                [0.5, 0.16666667, 0.0, -0.6666667, 0.0, 0.0],
                [0.16666667, 0.5, 0.0, -0.6666667, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-0.6666667, -0.6666667, 0.0, 1.3333334, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.3333334, -1.3333334],
                [0.0, 0.0, 0.0, 0.0, -1.3333334, 1.3333334],
            ],
            dtype=np.float32,
        )

        K_xy = np.array(
            [
                [0.5, 0.0, 0.16666667, 0.0, 0.0, -0.6666667],
                [0.16666667, 0.0, -0.16666667, -0.6666667, 0.6666667, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-0.6666667, 0.0, 0.0, 0.6666667, -0.6666667, 0.6666667],
                [0.0, 0.0, 0.6666667, -0.6666667, 0.6666667, -0.6666667],
                [0.0, 0.0, -0.6666667, 0.6666667, -0.6666667, 0.6666667],
            ],
            dtype=np.float32,
        )

        K_yy = np.array(
            [
                [0.5, 0.0, 0.16666667, 0.0, 0.0, -0.6666667],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.16666667, 0.0, 0.5, 0.0, 0.0, -0.6666667],
                [0.0, 0.0, 0.0, 1.3333334, -1.3333334, 0.0],
                [0.0, 0.0, 0.0, -1.3333334, 1.3333334, 0.0],
                [-0.6666667, 0.0, -0.6666667, 0.0, 0.0, 1.3333334],
            ],
            dtype=np.float32,
        )

        M = np.array(
            [
                [0.01666667, -0.00277778, -0.00277778, 0.0, -0.01111111, 0.0],
                [-0.00277778, 0.01666667, -0.00277778, 0.0, 0.0, -0.01111111],
                [-0.00277778, -0.00277778, 0.01666667, -0.01111111, 0.0, 0.0],
                [0.0, 0.0, -0.01111111, 0.08888889, 0.04444445, 0.04444445],
                [-0.01111111, 0.0, 0.0, 0.04444445, 0.08888889, 0.04444445],
                [0.0, -0.01111111, 0.0, 0.04444445, 0.04444445, 0.08888889],
            ],
            dtype=np.float32,
        )

        return [K_xx, K_xy, K_yy], M


def assembleSystemSequential(grid, K, M):
    """
    Assemble system matrix and right hand side with the help of the local matrices
    and the information stored in the grid.
    """
    numDofs = len(grid.dofs)
    systemMatrix = dok_matrix((numDofs, numDofs), dtype=np.float32)
    systemRightHand = np.zeros(numDofs, dtype=np.float32)

    for triangle in grid.triangles:
        # get all important triangle/ material parameters
        detJ, G = triangle.jacobi()
        # note: G = J^{-1} * J^{-T}
        a = triangle.material.get("a")
        c = triangle.material.get("c")
        f = triangle.material.get("f")
        numLocalDofs = len(triangle.dofs)

        # assemble cell matrix and cell right hand side
        cellMatrix = np.zeros((numLocalDofs, numLocalDofs), dtype=np.float32)
        cellRightHand = np.zeros(numLocalDofs, dtype=np.float32)

        # compute cellMatrix
        diffusionMatrix = a * (
            G[0, 0] * K[0] + G[0, 1] * K[1] + G[1, 0] * K[1].T + G[1, 1] * K[2]
        )
        reactionMatrix = c * M * detJ
        cellMatrix = diffusionMatrix + reactionMatrix

        # compute cellRightHand
        cellRightHand = f * detJ * np.sum(M, axis=1)

        # write local to global matrix
        for i, firstDof in enumerate(triangle.dofs):
            systemRightHand[firstDof.ind] += cellRightHand[i]
            for j, secondDof in enumerate(triangle.dofs):
                systemMatrix[firstDof.ind, secondDof.ind] += cellMatrix[i, j]

    return systemMatrix.tocsr(), systemRightHand


def assembleSystem(grid, K, M):
    """
    This is a multiprocessing version of assembleSystemSequential, if OS is Linux.
    """
    global processChunk
    import multiprocessing as mp
    import platform

    # if OS is NOT Linux, then fork and multiprocessing does not work properly
    if platform.system() != "Linux":
        return assembleSystemSequential(grid, K, M)

    # source (parallelization): https://datascience.blog.wzb.eu/2018/02/02/vectorization-and-parallelization-in-python-with-numpy-and-pandas/
    numCpu = mp.cpu_count()
    chunks = np.array_split(range(len(grid.triangles)), numCpu)

    numDofs = len(grid.dofs)
    systemMatrix = csr_matrix((numDofs, numDofs), dtype=np.float32)
    systemRightHand = np.zeros(numDofs, dtype=np.float32)

    # for triangle in grid.triangles:
    def processChunk(chunk):
        numDofs = len(grid.dofs)
        systemMatrixProc = dok_matrix((numDofs, numDofs), dtype=np.float32)
        systemRightHandProc = np.zeros(numDofs, dtype=np.float32)

        for i in chunk:
            triangle = grid.triangles[i]
            # get all important triangle/ material parameters
            detJ, G = triangle.jacobi()
            # note: G = J^{-1} * J^{-T}
            a = triangle.material.get("a")
            c = triangle.material.get("c")
            f = triangle.material.get("f")
            numLocalDofs = len(triangle.dofs)

            # assemble cell matrix and cell right hand side
            cellMatrix = np.zeros((numLocalDofs, numLocalDofs), dtype=np.float32)
            cellRightHand = np.zeros(numLocalDofs, dtype=np.float32)

            # compute cellMatrix
            diffusionMatrix = a * (
                G[0, 0] * K[0] + G[0, 1] * K[1] + G[1, 0] * K[1].T + G[1, 1] * K[2]
            )
            reactionMatrix = c * M * detJ
            cellMatrix = diffusionMatrix + reactionMatrix

            # compute cellRightHand
            cellRightHand = f * detJ * np.sum(M, axis=1)

            # write local to global matrix
            for i, firstDof in enumerate(triangle.dofs):
                systemRightHandProc[firstDof.ind] += cellRightHand[i]
                for j, secondDof in enumerate(triangle.dofs):
                    systemMatrixProc[firstDof.ind, secondDof.ind] += cellMatrix[i, j]

        return systemMatrixProc.tocsr(), systemRightHandProc

    with mp.Pool(processes=numCpu) as pool:
        procResults = [
            pool.apply_async(processChunk, args=(chunk,)) for chunk in chunks
        ]
        resultChunks = [r.get() for r in procResults]

    for matrix, rhs in resultChunks:
        systemMatrix += matrix
        systemRightHand += rhs

    return systemMatrix, systemRightHand


def applyBoundaryCondition(grid, matrix=None, vector=None, homogenize=False):
    """
    Apply Dirichlet or Neumann boundary conditions to a matrix and vector
    """
    for edge in grid.edges:
        if edge.boundaryConstraint is not None:
            if edge.boundaryConstraint.type == "Dirichlet":
                for dof in edge.dofs:
                    vector[dof.ind] = (
                        edge.boundaryConstraint.function(dof.x, dof.y)
                        if not homogenize
                        else 0.0
                    )
                    if matrix is not None:
                        # set row dof.ind to 0.0
                        matrix.data[
                            matrix.indptr[dof.ind] : matrix.indptr[dof.ind + 1]
                        ] = 0.0
                        matrix[dof.ind, dof.ind] = 1.0
            if edge.boundaryConstraint.type == "Neumann":
                pass


def getDirichletVector(grid):
    """
    Return a vector with components 1.0 if not a Dirchlet boundary DoF, else 0.0
    """
    dirichletVector = np.ones(len(grid.dofs), dtype=np.float32)

    for edge in grid.edges:
        if edge.boundaryConstraint is not None:
            if edge.boundaryConstraint.type == "Dirichlet":
                for dof in edge.dofs:
                    dirichletVector[dof.ind] = 0.0

    return dirichletVector
