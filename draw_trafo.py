import matplotlib.pyplot as plt
import numpy as np
plt.rc('text', usetex=True)
plt.rcParams["figure.figsize"] = (4,4)

def run():
    fig = plt.figure()

    ax1 = fig.add_subplot(1,2,1)
    xCoords, yCoords = [0.,0.,1.], [1.,0.,0.]
    ax1.plot([0.,0.,1.,0.], [1.,0.,0.,1.])
    ax1.scatter(xCoords, yCoords, color = "black")
    ax1.set_title("Master element")

    ax2 = fig.add_subplot(1,2,2)
    xCoords2, yCoords2 = [0.,0.5,0.5], [-1.,-1.,-0.5]
    ax2.plot([0.,0.5,0.5,0.], [-1.,-1.,-0.5,-1.])
    ax2.scatter(xCoords2, yCoords2, color = "black")
    ax2.set_title(r"$K_i$")
    #plt.savefig("triangle_trafo.svg", transparent = True)
    plt.show()

if __name__ == "__main__":
    run()
