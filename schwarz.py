from assembly import getLocalMatrices, assembleSystem , applyBoundaryCondition, getDirichletVector
from solver import Jacobi, ForwardGaussSeidel, BackwardGaussSeidel
from main import saveVtk, analyzeSolution

import time
import math
from math import log
import logging
from functools import wraps
import numpy as np
from scipy.sparse.linalg import spsolve, inv
from scipy.sparse import diags

# get current time in milliseconds
millis = lambda: int(round(time.time() * 1000))

# creating decorator to time functions
def timeit(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
        time_before = millis()
        try:
            return func(*args, **kwargs)
        finally:
            time_after = millis()
            time_delta = time_after - time_before
            logging.debug(f"{func.__name__} took {time_delta // 1000} seconds {time_delta % 1000} milliseconds.")
    return _time_it

class BPX():
    @timeit
    def __init__(self, coarseGrid, numberLevels = 2, showGrids = False):
        self.numberLevels = numberLevels
        self.prolongationMatrices = []
        self.dirichletVectors = []

        # 1. getLocalMatrices
        K , M = getLocalMatrices(degree = 1)

        # 2. getCoarseGrid
        self.grids = [coarseGrid]
        # which DoFs have Dirchlet BC?
        self.dirichletVectors.append(getDirichletVector(coarseGrid))
        # print and visualize coarse grid
        if showGrids:
            print(coarseGrid)
            coarseGrid.plot(title = "Coarse Grid")

        # 3. global grid refinement + assemble finer grid matrices
        for i in range(self.numberLevels - 1):
            # refine finest grid
            levelGrid = self.grids[-1].refine()

            # which DoFs have Dirchlet BC?
            self.dirichletVectors.append(getDirichletVector(levelGrid))

            # print and visualize level grid
            if showGrids:
                print(levelGrid)
                levelGrid.plot(title = f"Grid on level {i+1}")

            self.grids.append(levelGrid)

        # assemble level matrix and apply boundary conditions
        self.systemMatrix, self.systemRHS = assembleSystem(self.grids[-1], K, M)
        applyBoundaryCondition(self.grids[-1], self.systemMatrix, self.systemRHS)

        # compute D_L^{-1}
        diagonal = self.systemMatrix.diagonal() # diagonal vector from systemMatrix
        self.D_inverse = diags(1. / diagonal, format = "csr")

        if numberLevels > 1:
            logging.info(f"Number of DoFs: {len(self.grids[-1].dofs)} (by level: {','.join([str(len(g.dofs)) for g in self.grids])})")

    @timeit
    def buildTransfer(self):
        for i in range(self.numberLevels - 1):
            self.prolongationMatrices.append(
                self.grids[i+1].getInterpolationMatrix()
            )

    def addLevel(self, showGrids = False):
        self.numberLevels += 1

        # 1. getLocalMatrices
        K , M = getLocalMatrices(degree = 1)

        # 2. refine finest grid
        levelGrid = self.grids[-1].refine()

        # Which DoFs have Dirchlet BC?
        self.dirichletVectors.append(getDirichletVector(levelGrid))

        # 3. print and visualize level grid
        if showGrids:
            print(levelGrid)
            levelGrid.plot(title = f"Grid on level {i+1}")

        self.grids.append(levelGrid)

        # 4. assemble level matrix and apply boundary conditions
        self.systemMatrix, self.systemRHS = assembleSystem(levelGrid, K, M)
        applyBoundaryCondition(levelGrid, self.systemMatrix, self.systemRHS)

        # compute D_L^{-1}
        diagonal = self.systemMatrix.diagonal() # diagonal vector from systemMatrix
        self.D_inverse = diags(1. / diagonal, format = "csr")

        logging.info(f"Number of DoFs: {len(self.grids[-1].dofs)} (by level: {','.join([str(len(g.dofs)) for g in self.grids])})")

        # 5. build new interpolation matrix
        self.prolongationMatrices.append(
            self.grids[-1].getInterpolationMatrix()
        )

    @timeit
    def __call__(self, vector):
        vectors = [vector]

        # restriction
        for interpolationMatrix in self.prolongationMatrices[::-1]:
            vectors.append(interpolationMatrix.T.dot(vectors[-1]))
        vectors = vectors[::-1]

        # diagonal scaling
        for i, vector in enumerate(vectors):
            dim = vector.shape[0]
            vectors[i] = self.D_inverse[:dim,:dim].multiply(self.dirichletVectors[i]).dot(vector)

        # interpolation
        for i in range(1, len(vectors)):
            vectors[i] += self.prolongationMatrices[i-1].dot(vectors[i-1])

        return vectors[-1]

        """
        # the following code is only for testing
        # the iteration numbers match with the method above

        bigInterpolation = []
        for interpolationMatrix in self.prolongationMatrices[::-1]:
            if len(bigInterpolation) == 0:
                bigInterpolation.append(interpolationMatrix)
            else:
                bigInterpolation.append(bigInterpolation[-1].dot(interpolationMatrix))

        out = self.D_inverse.multiply(self.dirichletVectors[-1]).dot(vector)
        for interpolationMatrix in bigInterpolation:
            dim = interpolationMatrix.shape[1]
            out += interpolationMatrix.dot(self.D_inverse[:dim, :dim].multiply(self.dirichletVectors[-1][:dim])).dot(interpolationMatrix.T).dot(vector)

        return out
        """


class HB():
    @timeit
    def __init__(self, coarseGrid, numberLevels = 2, showGrids = False):
        self.numberLevels = numberLevels
        self.prolongationMatrices = []
        self.dirichletVectors = []

        # 1. getLocalMatrices
        K , M = getLocalMatrices(degree = 1)

        # 2. getCoarseGrid
        self.grids = [coarseGrid]
        # which DoFs have Dirchlet BC?
        self.dirichletVectors.append(getDirichletVector(coarseGrid))
        # print and visualize coarse grid
        if showGrids:
            print(coarseGrid)
            coarseGrid.plot(title = "Coarse Grid")

        # 3. global grid refinement + assemble finer grid matrices
        for i in range(self.numberLevels - 1):
            # refine finest grid
            levelGrid = self.grids[-1].refine()

            # which DoFs have Dirchlet BC?
            self.dirichletVectors.append(getDirichletVector(levelGrid))

            # print and visualize level grid
            if showGrids:
                print(levelGrid)
                levelGrid.plot(title = f"Grid on level {i+1}")

            self.grids.append(levelGrid)

        # assemble level matrix and apply boundary conditions
        self.systemMatrix, self.systemRHS = assembleSystem(self.grids[-1], K, M)
        applyBoundaryCondition(self.grids[-1], self.systemMatrix, self.systemRHS)

        # compute D_L^{-1}
        diagonal = self.systemMatrix.diagonal() # diagonal vector from systemMatrix
        self.D_inverse = diags(1. / diagonal, format = "csr")

        if numberLevels > 1:
            logging.info(f"Number of DoFs: {len(self.grids[-1].dofs)} (by level: {','.join([str(len(g.dofs)) for g in self.grids])})")

    @timeit
    def buildTransfer(self):
        for i in range(self.numberLevels - 1):
            self.prolongationMatrices.append(
                self.grids[i+1].getInterpolationMatrix()
            )

    def addLevel(self, showGrids = False):
        self.numberLevels += 1

        # 1. getLocalMatrices
        K , M = getLocalMatrices(degree = 1)

        # 2. refine finest grid
        levelGrid = self.grids[-1].refine()

        # Which DoFs have Dirchlet BC?
        self.dirichletVectors.append(getDirichletVector(levelGrid))

        # 3. print and visualize level grid
        if showGrids:
            print(levelGrid)
            levelGrid.plot(title = f"Grid on level {i+1}")

        self.grids.append(levelGrid)

        # 4. assemble level matrix and apply boundary conditions
        self.systemMatrix, self.systemRHS = assembleSystem(levelGrid, K, M)
        applyBoundaryCondition(levelGrid, self.systemMatrix, self.systemRHS)

        # compute D_L^{-1}
        diagonal = self.systemMatrix.diagonal() # diagonal vector from systemMatrix
        self.D_inverse = diags(1. / diagonal, format = "csr")

        logging.info(f"Number of DoFs: {len(self.grids[-1].dofs)} (by level: {','.join([str(len(g.dofs)) for g in self.grids])})")

        # 5. build new interpolation matrix
        self.prolongationMatrices.append(
            self.grids[-1].getInterpolationMatrix()
        )

    @timeit
    def __call__(self, vector):
        """
        vectors = [vector]

        newNodes = []
        grid = self.grids[-1]
        numDofs = len(grid.dofs)
        lvls = grid.dofs[-1].lvl
        lvlStart = [0]
        for dof in grid.dofs:
            if dof.lvl == len(lvlStart):
                lvlStart.append(dof.ind)
        lvlStart.append(grid.dofs[-1].ind+1)
        for i,start in enumerate(lvlStart[:-1]):
            vec = np.zeros(numDofs)
            for j in range(start,lvlStart[i+1]):
                vec[j] = 1.
            newNodes.append(vec)

        # restriction
        for interpolationMatrix in self.prolongationMatrices[::-1]:
            vectors.append(interpolationMatrix.T.dot(vectors[-1]))
        vectors = vectors[::-1]

        
        # diagonal scaling
        for i, vector in enumerate(vectors):
            dim = vector.shape[0]
            vectors[i] = self.D_inverse[:dim,:dim].multiply(newNodes[i][:dim]).multiply(self.dirichletVectors[i]).dot(vector)
        
        # interpolation
        for i in range(1, len(vectors)):
            vectors[i] += self.prolongationMatrices[i-1].dot(vectors[i-1])

        #return vectors[-1]
        """


        w = vector.copy()

        # restriction
        for i, interpolationMatrix in enumerate(self.prolongationMatrices[::-1]):
            n, m = interpolationMatrix.shape
            w[:m] = interpolationMatrix.T.dot(w[:n])
            

        # diagonal scaling
        w = self.D_inverse.multiply(self.dirichletVectors[-1]).dot(w)
        
        # interpolation
        for interpolationMatrix in self.prolongationMatrices:
            n, m = interpolationMatrix.shape
            w[m:n] += interpolationMatrix.dot(w[:m])[m:n]
            
        
        return w

        """
        # diagonal scaling
        w = self.D_inverse.multiply(self.dirichletVectors[-1]).dot(w)

        # interpolation
        for interpolationMatrix in self.prolongationMatrices:
            n, m = interpolationMatrix.shape
            w[:n] = interpolationMatrix.dot(w[:m])
            applyBoundaryCondition(grid = self.grids[-1], vector = w, homogenize = True)

        return w
        """

        # the following code is only for testing
        # the iteration numbers match with the method above
        """
        newNodes = []
        grid = self.grids[-1]
        numDofs = len(grid.dofs)
        lvls = grid.dofs[-1].lvl
        lvlStart = [0]
        for dof in grid.dofs:
            if dof.lvl == len(lvlStart):
                lvlStart.append(dof.ind)
        lvlStart.append(grid.dofs[-1].ind+1)
        for i,start in enumerate(lvlStart[:-1]):
            vec = np.zeros(numDofs)
            for j in range(start,lvlStart[i+1]):
                vec[j] = 1.
            newNodes.append(vec)

        bigInterpolation = []
        for interpolationMatrix in self.prolongationMatrices[::-1]:
            if len(bigInterpolation) == 0:
                bigInterpolation.append(interpolationMatrix)
            else:
                bigInterpolation.append(bigInterpolation[-1].dot(interpolationMatrix))

        n,m = self.prolongationMatrices[-1].shape
        out = self.D_inverse.multiply(newNodes[-1]).multiply(self.dirichletVectors[-1]).dot(vector)
        for interpolationMatrix, newNode in zip(bigInterpolation, newNodes[:-1][::-1]):
            dim = interpolationMatrix.shape[1]
            out += interpolationMatrix.dot(self.D_inverse[:dim, :dim].multiply(newNode[:dim]).multiply(self.dirichletVectors[-1][:dim])).dot(interpolationMatrix.T).dot(vector)

        return out
        """