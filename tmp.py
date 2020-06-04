from multigrid import Multigrid
from grid import homeworkGrid
from scipy.sparse.linalg import spsolve
from main import saveVtk
import logging

logging.basicConfig(level=logging.DEBUG , format='[%(asctime)s] - [%(levelname)s] - %(message)s')

coarseGrid = homeworkGrid()
mg = Multigrid(coarseGrid, numberLevels = 3)
mg.buildTransfer()
"""
print(mg.prolongationMatrices[0].todense())
print(mg.levelMatrices[0].todense())
print((mg.prolongationMatrices[0].T.dot(mg.levelMatrices[1].dot(mg.prolongationMatrices[0]))).todense())
print((mg.levelMatrices[0]-0.25*mg.prolongationMatrices[0].T.dot(mg.levelMatrices[1].dot(mg.prolongationMatrices[0]))).todense())
"""
print(mg.grids[0])
print((mg.levelMatrices[0]-0.25*mg.prolongationMatrices[0].T.dot(mg.levelMatrices[1].dot(mg.prolongationMatrices[0]))).todense())
#mg(maxIter = 1)

print(mg(maxIter = 100)[0]-spsolve(mg.levelMatrices[-1], mg.levelRHS[-1]))
#saveVtk(spsolve(mg.levelMatrices[-1], mg.levelRHS[-1]), mg.grids[-1])
