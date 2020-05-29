import numpy as np
from assembly import getLocalMatrices, assembleSystem , applyBoundaryCondition
from solver import Jacobi, ForwardGaussSeidel, BackwardGaussSeidel
from main import saveVtk, analyzeSolution
from scipy.sparse.linalg import spsolve

class Multigrid():
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

    def mgm(self, startVector, rightHandSide, level, mu, smoother, preSmooth, postSmooth, epsilon):
        # 1. Pre-smoothing
        solution, _ = smoother({"systemMatrix": self.levelMatrices[level],"rightHandSide": rightHandSide, "startVector": startVector, "maxIter": preSmooth, "epsilon": epsilon})

        # 2. Compute defect and restrict to level-1
        defect = self.restrict(self.defect(rightHandSide, solution, level), level)

        # 3. Coarse grid solution
        if level - 1 == 0:
            # direct solver on coarsest grid
            coarserSolution = spsolve(self.levelMatrices[0], defect)
        else:
            # level > 1
            for i in range(mu):
                coarserSolution = np.zeros(defect.shape)
                coarserSolution = self.mgm(
                    startVector = coarserSolution,
                    rightHandSide = defect,
                    level = level - 1,
                    mu = mu,
                    smoother = smoother,
                    preSmooth = preSmooth,
                    postSmooth = postSmooth,
                    epsilon = epsilon
                )

        # 4. update by interpolation of coarser grid solution
        solution += self.prolongate(coarserSolution, level)

        # 5. post smoothing
        solution, _ = smoother({"systemMatrix": self.levelMatrices[level],"rightHandSide": rightHandSide, "startVector": solution, "maxIter": postSmooth, "epsilon": epsilon})

        return solution

    def __call__(self, startVector = None, cycle = "V", smoother = lambda x: Jacobi()(**x, omega = 1.0), preSmooth = 1, postSmooth = 1, maxIter = 100, epsilon = 1e-12):
        # initialize additional variables
        if startVector is None:
            startVector = np.zeros(self.levelRHS[-1].shape)

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
            # solution accurate enough
            if np.linalg.norm(self.defect(self.levelRHS[-1], solution, self.numberLevels - 1)) < epsilon:
                iterations = iter
                break

            solution = self.mgm(solution, self.levelRHS[-1], self.numberLevels - 1, mu, smoother, preSmooth, postSmooth, epsilon)

        print(iter)
        print(solution)
        saveVtk(solution, self.grids[-1])

        return solution, iterations
