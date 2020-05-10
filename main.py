import numpy as np
from assembly import getLocalMatrices, assembleSystem , applyBoundaryCondition
from grid import homeworkGrid

GLOBAL_REFINEMENTS = 1
SHOW_GRIDS = True
DEGREE = 1

if __name__ == "__main__":
    # 1. getLocalMatrices
    K , M = getLocalMatrices(degree = DEGREE)
    # TODO: Debug degree = 2,3 !!!!

    # 2. getCoarseGrid
    coarseGrid = homeworkGrid(degree = DEGREE)
    if SHOW_GRIDS:
        print(coarseGrid)
        coarseGrid.plot(title = "Coarse Grid")

    # 3. assemble, apply BC
    coarseMatrix, coarseRHS = assembleSystem(coarseGrid, K, M)
    print(coarseMatrix.todense())
    print(coarseRHS)
    coarseMatrix_new, coarseRHS_new = applyBoundaryCondition(coarseGrid,coarseMatrix,coarseRHS)
    print(coarseMatrix.todense())
    print(coarseRHS)
    print((coarseMatrix-coarseMatrix_new).todense())
    print((coarseRHS_new-coarseRHS))  #don't know why this is array of zeros
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
        # TODO

    # 5. Multigrid ...
