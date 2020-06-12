from assembly import getLocalMatrices, assembleSystem , applyBoundaryCondition
from solver import Jacobi, ForwardGaussSeidel, BackwardGaussSeidel
from main import saveVtk, analyzeSolution

import time
import logging
from functools import wraps
import numpy as np
from scipy.sparse.linalg import spsolve, inv

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

class Multigrid():
    @timeit
    def __init__(self, coarseGrid, numberLevels = 2, showGrids = False):
        self.numberLevels = numberLevels

        # 1. getLocalMatrices
        K , M = getLocalMatrices(degree = 1)

        # 2. getCoarseGrid
        self.grids = [coarseGrid]
        if showGrids:
            print(coarseGrid)
            coarseGrid.plot(title = "Coarse Grid")

        # 3. assemble, apply BC
        coarseMatrix, coarseRHS = assembleSystem(coarseGrid, K, M)
        applyBoundaryCondition(coarseGrid,coarseMatrix,coarseRHS)

        self.levelMatrices = [coarseMatrix]
        self.levelRHS      = [coarseRHS]

        # 4. global grid refinement + assemble finer grid matrices
        for i in range(self.numberLevels - 1):
            # refine finest grid
            levelGrid = self.grids[-1].refine()

            # print and visualize level grid
            if showGrids:
                print(levelGrid)
                levelGrid.plot(title = f"Grid on level {i+1}")

            self.grids.append(levelGrid)

            # assemble level matrix and apply boundary conditions
            levelMatrix, levelRHS = assembleSystem(levelGrid, K, M)
            applyBoundaryCondition(levelGrid,levelMatrix,levelRHS)

            self.levelMatrices.append(levelMatrix)
            self.levelRHS.append(levelRHS)

        logging.info(f"Number of DoFs: {len(self.grids[-1].dofs)} (by level: {','.join([str(len(g.dofs)) for g in self.grids])})")

    @timeit
    def buildTransfer(self):
        self.prolongationMatrices = []
        for i in range(self.numberLevels - 1):
            self.prolongationMatrices.append(
                self.grids[i+1].getInterpolationMatrix()
            )

    def prolongate(self,vector,toLvl):
        return self.prolongationMatrices[toLvl - 1].dot(vector)

    def restrict(self,vector,fromLvl):
        return 0.25 * self.prolongationMatrices[fromLvl - 1].T.dot(vector)

    def defect(self, rightHandSide, solution, lvl):
        return rightHandSide - self.levelMatrices[lvl].dot(solution)

    def mgm(self, startVector, rightHandSide, level, mu, preSmoother, postSmoother, preSmoothSteps, postSmoothSteps, epsilon):
        # 1. Pre-smoothing
        solution, _ = preSmoother({"systemMatrix": self.levelMatrices[level],"rightHandSide": rightHandSide, "startVector": startVector, "maxIter": preSmoothSteps, "epsilon": epsilon})

        # 2. Compute defect and restrict to level-1
        defect = self.restrict(self.defect(rightHandSide, solution, level), level)
        applyBoundaryCondition(grid = self.grids[level - 1], vector = defect, homogenize = True)

        # 3. Coarse grid solution
        if level - 1 == 0:
            # direct solver on coarsest grid
            coarserSolution = inv(self.levelMatrices[0].tocsc()).dot(defect)
        else:
            # level > 1
            coarserSolution = np.zeros(defect.shape)
            for i in range(mu):
                coarserSolution = self.mgm(
                    startVector = coarserSolution,
                    rightHandSide = defect,
                    level = level - 1,
                    mu = mu,
                    preSmoother = preSmoother,
                    postSmoother = postSmoother,
                    preSmoothSteps = preSmoothSteps,
                    postSmoothSteps = postSmoothSteps,
                    epsilon = epsilon
                )

        # 4. update by interpolation of coarser grid solution
        update = self.prolongate(coarserSolution, level)
        applyBoundaryCondition(grid = self.grids[level], vector = update, homogenize = True)
        solution += update

        # 5. post smoothing
        solution, _ = postSmoother({"systemMatrix": self.levelMatrices[level],"rightHandSide": rightHandSide, "startVector": solution, "maxIter": postSmoothSteps, "epsilon": epsilon})

        return solution

    @timeit
    def __call__(self, startVector = None, cycle = "W", preSmoother = lambda x: ForwardGaussSeidel()(**x), postSmoother = lambda x: BackwardGaussSeidel()(**x), preSmoothSteps = 2, postSmoothSteps = 2, maxIter = 100, epsilon = 1e-12):
        # initialize additional variables
        if startVector is None:
            startVector = np.zeros(self.levelRHS[-1].shape)
            applyBoundaryCondition(grid = self.grids[-1], vector = startVector, homogenize = False)

        mu = -1
        if cycle == "V":
            mu = 1
        elif cycle == "W":
            mu = 2
        else:
            raise Exception(f"{cycle} cycle is currently not supported!")

        solution  = startVector
        iterations = maxIter

        for iter in range(maxIter):
            # log defect before iteration
            normOfDefect = np.linalg.norm(self.defect(self.levelRHS[-1], solution, self.numberLevels - 1))
            logging.debug(f"{iter}.th defect: {normOfDefect}")

            # solution accurate enough
            if  normOfDefect < epsilon:
                iterations = iter
                break

            solution = self.mgm(solution, self.levelRHS[-1], self.numberLevels - 1, mu, preSmoother, postSmoother, preSmoothSteps, postSmoothSteps, epsilon)

        logging.info(f"GMG iterations: {iterations}")
        saveVtk(solution, self.grids[-1])

        return solution, iterations
