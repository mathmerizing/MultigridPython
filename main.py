from assembly import getLocalMatrices, assembleSystem , applyBoundaryCondition
from grid import homeworkGrid, unitSquare
from solver import Jacobi, ForwardGaussSeidel, BackwardGaussSeidel

import sys
import logging
import numpy as np

parameters = {
    "LEVELS": 5,
    "SHOW_GRIDS": False,
    "DEGREE": 1,
    "CYCLE": "W",
    "MAX_ITER": 100,
    "SMOOTHING_STEPS": 2,
    "SMOOTHER": "Jacobi",
    "OMEGA": 0.8
}

def getParam(userInput, default):
    if userInput == "":
        return default
    else:
        return userInput

def inputParameters():
    print("==================")
    print("* DEFAULT VALUES *")
    print("==================")

    for key in parameters:
        print(f"{key}{' '*(20-len(key))}= {parameters[key]}")
    print("")

    change = input("Would you like to change some values? (Y/N): ")
    if change in ["Y","y","yes", "YES", "Yes", "ja", "Ja", "JA"]:
        # manually change all parameters
        lvls = parameters["LEVELS"]
        parameters["LEVELS"] = int(getParam(input(f"Number of MG levels (default: {lvls}) = "),lvls))

        show = parameters["SHOW_GRIDS"]
        parameters["SHOW_GRIDS"] = bool(getParam(input(f"Plot grids (default: {show}) = "),show))

        deg = parameters["DEGREE"]
        parameters["DEGREE"] = int(getParam(input(f"Degree of FE (default: {deg}) = "),deg))

        cyc = parameters["CYCLE"]
        parameters["CYCLE"] = getParam(input(f"Multigrid cycle (default: {cyc}; supported: 'V','W') = "),cyc)

        it = parameters["MAX_ITER"]
        parameters["MAX_ITER"] = int(getParam(input(f"Maximum number of MG iterations (default: {it}) = "),it))

        st = parameters["SMOOTHING_STEPS"]
        parameters["SMOOTHING_STEPS"] = int(getParam(input(f"Number of smoothing steps (default: {st}) = "),st))

        sm = parameters["SMOOTHER"]
        parameters["SMOOTHER"] = getParam(input(f"Smoother type (default: {sm}; supported: 'Jacobi','GaussSeidel') = "),sm)

        if parameters["SMOOTHER"] == "Jacobi":
            om = parameters["OMEGA"]
            parameters["OMEGA"] = float(getParam(input(f"Relaxation parameter omega (default: {om}) = "),om))

        print("")
        print("=================")
        print("* CUSTOM VALUES *")
        print("=================")
        for key in parameters:
            print(f"{key}{' '*(20-len(key))}= {parameters[key]}")
        print("")

def run():
    from multigrid import Multigrid
    logging.info("Starting run method...")
    coarseGrid = homeworkGrid()
    mg         = Multigrid(coarseGrid, numberLevels = parameters["LEVELS"], showGrids = parameters["SHOW_GRIDS"])
    mg.buildTransfer() # compute transfer matrices

    if parameters["SMOOTHER"] == "GaussSeidel":
        PRE_SMOOTHER  = lambda x: ForwardGaussSeidel()(**x)
        POST_SMOOTHER = lambda x: BackwardGaussSeidel()(**x)
    else:
        # parameters["SMOOTHER"] == "Jacobi"
        PRE_SMOOTHER  = lambda x: Jacobi()(**x, omega = parameters["OMEGA"])
        POST_SMOOTHER = lambda x: Jacobi()(**x, omega = parameters["OMEGA"])

    # run MG algo
    mg(
        maxIter = parameters["MAX_ITER"],
        cycle = parameters["CYCLE"],
        preSmoother = PRE_SMOOTHER,
        postSmoother = POST_SMOOTHER,
        preSmoothSteps = parameters["SMOOTHING_STEPS"],
        postSmoothSteps = parameters["SMOOTHING_STEPS"]
    )


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

def saveVtk(solution, grid, fileName = "solution.vtk"):
    lines = ["# vtk DataFile Version 3.0", "PDE solution", "ASCII", "DATASET UNSTRUCTURED_GRID"]

    lines.append(f"POINTS {len(grid.dofs)} double")
    for dof in grid.dofs:
        lines.append(f"{dof.x} {dof.y} 0.0")

    basisSize = 3 if grid.degree == 1 else 6
    cellType = 5 if grid.degree == 1 else 22
    numTriangles = len(grid.triangles)

    lines.append(f"CELLS {numTriangles} {numTriangles * (basisSize + 1)}")
    for triangle in grid.triangles:
        lines.append(f"{basisSize} {' '.join([str(dof.ind) for dof in triangle.dofs])}")

    lines.append(f"CELL_TYPES {numTriangles}")
    lines += numTriangles * [f"{cellType}"]

    lines.append(f"POINT_DATA {len(grid.dofs)}")
    lines.append("SCALARS u(x,y) double 1")
    lines.append("LOOKUP_TABLE default")

    for i in range(len(grid.dofs)):
        lines.append(str(solution[i]))

    with open(fileName, "w") as file:
        file.write("\n".join(lines))


def analyzeSolution(solution, iter, grid, matrix, rhs):
    print("*** ANALYSIS OF SOLUTION OF ITERATIVE SOLVER: ***")
    print("Iterations:", iter)
    print("Solution of iterative solver:\n", solution)
    print("Residual of solution:\n", np.linalg.norm(rhs - matrix.dot(solution)))

    print()
    print("Solution at each DoF:")
    for dof in grid.dofs:
        print(f"({dof.x},{dof.y}): {solution[dof.ind]}")
    print()

    exactSolution = np.dot(np.linalg.inv(matrix.todense()),rhs)
    print("exactSolution:\n",exactSolution)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG , format='[%(asctime)s] - [%(levelname)s] - %(message)s')
    inputParameters()
    run()
