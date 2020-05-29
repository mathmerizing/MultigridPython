from multigrid import Multigrid
from grid import homeworkGrid

coarseGrid = homeworkGrid()
mg = Multigrid(coarseGrid, numberLevels = 3)
mg.buildTransfer()
"""
print(mg.prolongationMatrices[0].todense())
print(mg.levelMatrices[0].todense())
print((mg.prolongationMatrices[0].T.dot(mg.levelMatrices[1].dot(mg.prolongationMatrices[0]))).todense())
print((mg.levelMatrices[0]-0.25*mg.prolongationMatrices[0].T.dot(mg.levelMatrices[1].dot(mg.prolongationMatrices[0]))).todense())
"""
mg(maxIter = 1000)
