import matplotlib.pyplot as plt
import numpy as np

class Node():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"({self.x},{self.y})"

    def dist(self, other):
        return np.sqrt((self.x-other.x)**2 + (self.y-other.y)**2)

    def midpoint(self,other):
        return Node(0.5*(self.x+other.x),0.5*(self.y+other.y))

class Edge():
    def __init__(self, start, end, boundaryConstraint = None):
        self.start = start
        self.end   = end

        self.boundaryConstraint = boundaryConstraint

    def nodes(self):
        return [self.start, self.end]

    def __str__(self):
        boundaryInfo = " "
        if self.boundaryConstraint != None:
            boundaryInfo += "BC: " + str(self.boundaryConstraint)
        return str(self.start) + " -> " + str(self.end) + boundaryInfo

class Material():
    def __init__(self, values = {}):
        self.values = values

    def __str__(self):
        out = []
        for key in self.values:
            out.append(f"{key}: {str(self.values[key])}")
        return ";".join(out)

class Triangle():
    def __init__(self, nodes, edges, material = Material()):
        self.nodes    = nodes
        self.edges    = edges
        self.material = material

    def __str__(self):
        out = ["TRIANGLE:"]
        # nodes
        out.append("-----> NODES:")
        for i, node in enumerate(self.nodes):
            out.append(f"\t {i+1}. {str(node)}")

        # egdes
        out.append("-----> EDGES:")
        for i, edge in enumerate(self.edges):
            out.append(f"\t {i+1}. {str(edge)}")

        # material
        out.append(f"-----> MATERIAL: {str(self.material)}")
        out.append("")
        return "\n".join(out)

class BoundaryCondition():
    def __init__(self, type = "Dirichlet", function = lambda x,y: 0.0):
        self.type     = type
        self.function = function

    def __str__(self):
        return self.type

class Grid():
    def __init__(self, nodes, edges, triangles):
        self.nodes     = nodes
        self.edges     = edges
        self.triangles = triangles

    def getActiveNodes(self):
        return len(self.nodes)

    def plot(self):
        # plot edges
        for i,edge in enumerate(self.edges):
            n1, n2 = edge.start, edge.end
            plt.plot([n1.x,n2.x], [n1.y,n2.y], color = "blue")
            middle = n1.midpoint(n2)
            plt.text(middle.x, middle.y, str(i+1), color = "blue")

        # plot nodes
        xCoords, yCoords = [], []
        for i,node in enumerate(self.nodes):
            xCoords.append(node.x)
            yCoords.append(node.y)
            plt.text(node.x,node.y, str(i+1))
        plt.scatter(xCoords, yCoords, color = "black")

        # label the triangles
        for i, triangle in enumerate(self.triangles):
            n1, n2, n3 = triangle.nodes
            middle = n1.midpoint(n2.midpoint(n3))
            plt.text(middle.x, middle.y, str(i+1), color = "red")

        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def __str__(self):
        out = []
        out.append("-"*20 + " GRID " + "-"*20)

        # nodes
        out.append("--> NODES:")
        for i, node in enumerate(self.nodes):
            out.append(f"\t {i+1}. {str(node)}")

        # egdes
        out.append("--> EDGES:")
        for i, edge in enumerate(self.edges):
            out.append(f"\t {i+1}. {str(edge)}")

        # triangles
        out.append("--> TRIANGLES:")
        for i, triangle in enumerate(self.triangles):
            out.append(f"\t {i+1}. {str(triangle)}")

        out.append("-"*46)
        return "\n".join(out)

def homeworkGrid():
    # create nodes of L-shape
    nodes = []
    for x in [-1.,0.,1.]:
        for y in [-1.,0.,1.]:
            if x != 1. or y != 1.:
                nodes.append(Node(x,y))

    # define edges
    edges = []
    for i, firstNode in enumerate(nodes):
        for j, secondNode in enumerate(nodes):
            if firstNode.dist(secondNode) < 1.1 and i > j:
                edges.append(Edge(firstNode, secondNode))

    # diagonal edges
    edges.append(Edge(nodes[0], nodes[4]))
    edges.append(Edge(nodes[3], nodes[7]))
    edges.append(Edge(nodes[1], nodes[5]))

    # add BoundaryCondition to edges
    dirichletBoundaryConditions = BoundaryCondition("Dirichlet")
    neumannBoundaryConditions = BoundaryCondition("Neumann")

    edges[6].boundaryConstraint = dirichletBoundaryConditions
    edges[8].boundaryConstraint = dirichletBoundaryConditions

    for i in [0,1,2,5,7,9]:
        edges[i].boundaryConstraint = neumannBoundaryConditions

    # define triangles
    triangles = []
    for i, firstEdge in enumerate(edges):
        for j, secondEdge in enumerate(edges):
            for k, thirdEdge in enumerate(edges):
                # we don't want duplicate elements
                if i > j and j > k:
                    # is this actually a triangle ?
                    triangleNodes = set(firstEdge.nodes() + secondEdge.nodes() + thirdEdge.nodes())
                    if len(triangleNodes) == 3:
                        triangles.append(
                            Triangle(
                                list(triangleNodes),
                                [firstEdge, secondEdge, thirdEdge]
                            )
                        )

    my_material = lambda val : Material({"a": 1., "c": 0., "f":  val})

    # set materials of triangles
    triangles[0].material = my_material(0.)
    triangles[1].material = my_material(0.)

    triangles[2].material = my_material(1.)
    triangles[3].material = my_material(1.)

    triangles[4].material = my_material(-1.)
    triangles[5].material = my_material(-1.)

    return Grid(nodes, edges, triangles)
