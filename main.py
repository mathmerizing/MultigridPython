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

    B = loadMatlabMatrix("matlab_matrix.txt",4)
    matricesPermutationEquivalent(coarseMatrix.todense(),B)
    quit()

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

def matricesPermutationEquivalent(A,B):
    from itertools import permutations

    bestPermutation = None
    bestComponentDifference = 10**10

    for p in permutations(range(A.shape[0])):
        p = np.array(p)
        if np.max(A[p][:,p]-B) < bestComponentDifference:
            bestPermutation = p
            bestComponentDifference = np.max(A[p][:,p]-B)

    print("\nFirst matrix:")
    print(A)

    print("\nSecond matrix:")
    print(B)

    print("\nPermutation:", bestPermutation)
    print("Biggest difference:", bestComponentDifference)

    print("\nPermuted first matrix:")
    print(A[bestPermutation][:, bestPermutation])

    print("\nDifference:")
    print(A[bestPermutation][:, bestPermutation] - B)

def loadMatlabMatrix(txtFile, dim):
    matrix = np.zeros((dim,dim), dtype=np.float32)
    with open(txtFile, "r") as file:
        for i, line in enumerate(file):
            line = line.strip("\n").lstrip(" ")
            values = [np.float32(val.strip(" "))  for val in line.split(" ") if val != ""]
            matrix[i,:] = values
    return matrix

if __name__ == "__main__":
    paramterList = sys.argv[1:]
    if (len(paramterList) > 0):
        parseParameters(paramterList)

    # run FEM program with Multigrid preconditioner
    run()
