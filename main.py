import numpy as np
from assembly import getLocalMatrices, assembleSystem
from grid import homeworkGrid

GLOBAL_REFINEMENTS = 1
SHOW_GRIDS = False

if __name__ == "__main__":
    # 1. getLocalMatrices
    K , M = getLocalMatrices(degree = 1)
    # TODO: Debug degree = 2,3 !!!!

    # 2. getCoarseGrid
    coarseGrid = homeworkGrid()
    if SHOW_GRIDS:
        print(coarseGrid)
        coarseGrid.plot(title = "Coarse Grid")

    # 3. assemble, apply BC
    coarseMatrix, coarseRHS = assembleSystem(coarseGrid, K, M)
    print(coarseMatrix.todense())
    print(coarseRHS)
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
