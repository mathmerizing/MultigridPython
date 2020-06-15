import matplotlib.pyplot as plt
import numpy as np
plt.rc('text', usetex=True)

def run():
    #title = 'Domain'

    # edges
    plt.plot([0.,-1.,-1.,-1.,0.,1.,1.], [1.,1.,0.,-1.,-1.,-1.,0.], color = "blue", label = "homogeneous Neumann BC")
    plt.plot([1.,0.,0.], [0.,0.,1.], color = "red", label = "homogeneous Dirichlet BC")
    plt.plot([-1.,0.,0.], [0.,0.,-1.], "--", color = "black")

    """
    xCoords, yCoords = [], []
    for x in [-1.,0.,1.]:
        for y in [-1.,0.,1.]:
            if x != 1. or y != 1.:
                xCoords.append(x)
                yCoords.append(y)
    plt.scatter(xCoords, yCoords, color = "black")
    """

    plt.fill([-1.,-1.,0.,0.], [0.,1.,1.,0.], color='blue', alpha=0.2)
    plt.text(-0.5,0.5, r"$f = -1$")

    plt.fill([-1.,-1.,0.,0.], [-1.,0.,0.,-1.], color='orange', alpha=0.2)
    plt.text(-0.5,-0.5, r"$f = 0$")

    plt.fill([0.,0.,1.,1.], [-1.,0.,0.,-1.], color='green', alpha=0.2)
    plt.text(0.5,-0.5, r"$f = 1$")

    #plt.title(title)

    plt.legend(loc='upper right', shadow = True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("domain.svg", transparent = True)
    plt.show()

if __name__ == "__main__":
    run()
