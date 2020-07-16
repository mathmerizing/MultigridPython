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
