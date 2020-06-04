from scipy.sparse import diags
import numpy as np
from abc import ABC, abstractmethod

class Solver(ABC):
    """
    An abstract class which represents the solver of a system of linear equations.
    """

    @abstractmethod
    def __call__(self, systemMatrix, rightHandSide, startVector, maxIter, epsilon):
        """
        Solve the linear equation system with systemMatrix and rightHandSide.
        Use the startVector and use up to maxIter iterations or approximate
        up to an accuracy of epsilon.
        """
        pass


class Jacobi(Solver):
    def __call__(self, systemMatrix, rightHandSide, startVector, maxIter, epsilon, omega):
        diagonal = systemMatrix.diagonal() # diagonal vector from systemMatrix
        D_inverse = omega * diags(1. / diagonal, format = "csr")

        solution = startVector.copy()

        for iter in range(maxIter):
            solution += D_inverse.dot(rightHandSide - systemMatrix.dot(solution))

            if np.linalg.norm(rightHandSide - systemMatrix.dot(solution)) < epsilon:
                return solution, iter + 1

        return solution, maxIter


class ForwardGaussSeidel(Solver):
    def __call__(self, systemMatrix, rightHandSide, startVector, maxIter, epsilon):
        diagonal = systemMatrix.diagonal() # diagonal vector from systemMatrix
        solution = startVector.copy()
        dim = solution.shape[0]

        for iter in range(maxIter):
            for i in range(dim):
                row = systemMatrix.getrow(i)
                solution[i] = 0.
                solution[i] = (rightHandSide[i] - row.dot(solution)) / diagonal[i]

            if np.linalg.norm(rightHandSide - systemMatrix.dot(solution)) < epsilon:
                return solution, iter + 1

        return solution, maxIter


class BackwardGaussSeidel(Solver):
    def __call__(self, systemMatrix, rightHandSide, startVector, maxIter, epsilon):
        diagonal = systemMatrix.diagonal() # diagonal vector from systemMatrix
        solution = startVector.copy()
        dim = solution.shape[0]

        for iter in range(maxIter):
            for i in range(dim-1,-1,-1):
                row = systemMatrix.getrow(i)
                solution[i] = 0.
                solution[i] = (rightHandSide[i] - row.dot(solution)) / diagonal[i]

            if np.linalg.norm(rightHandSide - systemMatrix.dot(solution)) < epsilon:
                return solution, iter + 1

        return solution, maxIter
