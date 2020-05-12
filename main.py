import numpy as np
from assembly import getLocalMatrices, assembleSystem , applyBoundaryCondition
from grid import homeworkGrid, unitSquare
import sys

GLOBAL_REFINEMENTS = 1
SHOW_GRIDS         = True
DEGREE             = 1

def parseParameters(paramterList):
    for i, parameter in enumerate(paramterList):
        global GLOBAL_REFINEMENTS, SHOW_GRIDS, DEGREE

        if "--" not in parameter:
            continue
        # parse flag and parameters
        if "refinements" in parameter:
            GLOBAL_REFINEMENTS = int(paramterList[i+1])
        if "visualize" in parameter:
            SHOW_GRIDS = True
        if "release" in parameter:
            SHOW_GRIDS = False
        if "degree" in parameter:
            DEGREE = int(paramterList[i+1])

def run():
    # 1. getLocalMatrices
    K , M = getLocalMatrices(degree = DEGREE)
    # TODO: Debug degree = 2,3 !!!!

    # 2. getCoarseGrid
    #coarseGrid = homeworkGrid(degree = DEGREE)
    coarseGrid = unitSquare(degree = DEGREE)
    if SHOW_GRIDS:
        print(coarseGrid)
        coarseGrid.plot(title = "Coarse Grid")

    # 3. assemble, apply BC
    coarseMatrix, coarseRHS = assembleSystem(coarseGrid, K, M)
    applyBoundaryCondition(coarseGrid,coarseMatrix,coarseRHS)

    print(coarseMatrix.todense())
    print(coarseRHS)
    print(np.dot(np.linalg.inv(coarseMatrix.todense()),coarseRHS))
    quit()

    # 4. global grid refinement + assemble finer grid matrices
    grids = [coarseGrid]
    for i in range(GLOBAL_REFINEMENTS):
        # refine finest grid
        levelGrid = grids[-1].refine()

        # print and visualize level grid
        if SHOW_GRIDS:
            print(levelGrid)
            levelGrid.plot(title = f"Grid on level {i+1}")

        grids.append(levelGrid)

        # assemble level matrix and apply boundary conditions
        levelMatrix, levelRHS = assembleSystem(levelGrid, K, M)
        levelMatrix, levelRHS = applyBoundaryCondition(levelGrid,levelMatrix,levelRHS)
        print(levelMatrix.todense())
        print(levelRHS)
    # 5. Multigrid ...

if __name__ == "__main__":
    paramterList = sys.argv[1:]
    if (len(paramterList) > 0):
        parseParameters(paramterList)

    # run FEM program with Multigrid preconditioner
    run()
