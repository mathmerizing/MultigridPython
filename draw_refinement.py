import matplotlib.pyplot as plt
import numpy as np
plt.rc('text', usetex=True)
plt.rcParams["figure.figsize"] = (4,4)

def run():
    xCoords, yCoords = [0.,0.,2.], [2.,0.,0.]
    plt.plot([0.,0.,2.,0.], [2.,0.,0.,2.])
    plt.scatter(xCoords, yCoords, color = "black")

    i = 0
    for x,y in zip(xCoords, yCoords):
        i += 1
        plt.text(x+0.04,y+0.04, str(i), color = "black")

    xCoords2, yCoords2 = [0.,1.,1.], [1.,1.,0.]
    plt.plot([0.,1.,1.,0.], [1.,1.,0.,1.], "--", color = "green")
    plt.scatter(xCoords2, yCoords2, color = "green")

    for x,y in zip(xCoords2, yCoords2):
        i += 1
        plt.text(x+0.04,y+0.04, str(i), color = "green")

    plt.axis("off")

    plt.savefig("triangle_refinement.svg", transparent = True)
    plt.show()

if __name__ == "__main__":
    run()
