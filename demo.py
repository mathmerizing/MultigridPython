from main import inputParametersBPX, runDemoBPX, runDemoHB
import logging
logging.basicConfig(level=logging.INFO , format='[%(asctime)s] - [%(levelname)s] - %(message)s')

inputParametersBPX()

#Now that the user has inserted the parameters of the multigrid algorithm,
#we start by solving the two-grid method, refine the grid, solve the three-grid method, etc.

# normal Demo all levels
print("\n" + "="*35)
print("* BPX with 0 initial refinements: *".upper())
print("="*35 + "\n")
runDemoBPX(initialRefinements = 0)

print("\n" + "="*34)
print("* HB with 0 initial refinements: *".upper())
print("="*34 +"\n")
runDemoHB(initialRefinements = 0)

# Demo with coarse grid solver
print("\n" + "="*35)
print("* BPX with 4 initial refinements: *".upper())
print("="*35 + "\n")
runDemoBPX(initialRefinements = 4)

print("\n" + "="*34)
print("* HB with 4 initial refinements: *".upper())
print("="*34 +"\n")
runDemoHB(initialRefinements = 4)
