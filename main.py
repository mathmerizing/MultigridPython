import numpy as np
from assembly import getLocalMatrices
from grid import homeworkGrid

GLOBAL_REFINEMENTS = 1;




if __name__ == "__main__":
    # 1. getLocalMatrices
    K , M = getLocalMatrices(degree = 1)
    # TODO: Debug degree = 2,3 !!!!

    # 2. getCoarseGrid
    grid = homeworkGrid()
    print(grid)
    grid.plot()

    # 3. assemble, apply BC

    # 4. global grid refinement + assembly finer grid matrices

    # 5. Multigrid ...
