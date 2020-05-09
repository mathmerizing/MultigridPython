import matplotlib.pyplot as plt
import numpy as np

class Node():
    def __init__(self, x, y, ind = -1):
        self.x = x
        self.y = y
        self.ind = ind

    def __str__(self):
        return f"({self.x},{self.y}) [{self.ind+1}]"

    def dist(self, other):
        return np.sqrt((self.x-other.x)**2 + (self.y-other.y)**2)

    def midpoint(self,other):
        return Node(0.5*(self.x+other.x),0.5*(self.y+other.y))

class Edge():
    def __init__(self, start, end, boundaryConstraint = None):
        self.start = start
        self.end   = end

        # parameters for refinement
        self.children = []
        self.middle   = None

        self.boundaryConstraint = boundaryConstraint

    def length(self):
        return self.end.dist(self.start)

    def orientedVector(self, startNode):
        if startNode == self.start:
            return self.vector()
        else:
            return -1. * self.vector()

    def vector(self):
        return np.array([
            self.end.x-self.start.x,
            self.end.y-self.start.y
        ])

    def nodes(self):
        return [self.start, self.end]

    def commonNode(self,other):
        if self.start in other.nodes():
            return self.start
        else:
            return self.end

    def getChildren(self):
        if len(self.children) == 0:
            self.middle = self.start.midpoint(self.end)
            leftEdge  = Edge(self.start,  self.middle, self.boundaryConstraint)
            rightEdge = Edge(self.middle, self.end,    self.boundaryConstraint)
            self.children = [leftEdge, rightEdge]
        return self.children, self.middle

    def __str__(self):
        boundaryInfo = " "
        if self.boundaryConstraint != None:
            boundaryInfo += "BC: " + str(self.boundaryConstraint)
        return str(self.start) + " -> " + str(self.end) + boundaryInfo

class Material():
    def __init__(self, values = {}):
        self.values = values

    def get(self, variable):
        return self.values[variable]

    def __str__(self):
        out = []
        for key in self.values:
            out.append(f"{key}: {str(self.values[key])}")
        return ";".join(out)

class Triangle():
    def __init__(self, nodes, edges, material = Material()):
        self.nodes = nodes
        # sort the edges such that self.nodes[i] doesn't lie on the edge self.edges[i]
        self.edges = []
        for node in self.nodes:
            for edge in edges:
                if node not in edge.nodes():
                    self.edges.append(edge)

        self.material = material
        self.dofs = self.nodes[:]

    def distributeDofs(self, degree):
        if degree == 3:
            raise NotImplementedError()

        if degree == 2:
            for edge in self.edges:
                _, midpoint = edge.getChildren()
                self.dofs.append(midpoint)

    def jacobi(self):
        # compute jacobi matrix of the transformation from the
        # reference triangle
        hypothenuseLength = max([e.length() for e in self.edges])
        shortEdges = []
        jacobiMatrix = np.zeros((2,2),dtype=np.float32)
        for edge in self.edges:
            if edge.length() < hypothenuseLength:
                # edge is not the hypothenuse
                shortEdges.append(edge)
        # determine the node at the right angle of the triangle
        startNode = shortEdges[0].commonNode(shortEdges[1])
        # fill jacobi matrix
        jacobiMatrix[:,0] = shortEdges[0].orientedVector(startNode)
        jacobiMatrix[:,1] = shortEdges[1].orientedVector(startNode)

        # compute determinant of jacobi matrix
        jacobiDeterminant = np.linalg.det(jacobiMatrix)

        # compute J^{-1} * J^{-T} where J is the jacobi matrix
        jacobiInverse = np.linalg.inv(jacobiMatrix)

        return abs(jacobiDeterminant), np.dot(jacobiInverse, jacobiInverse.T)


    def jacobiDeterminant(self):
        # compute determinant of jacobi matrix
        if self.determinant == None:
            # compute the jacobi determinant of the transformation from the
            # reference triangle
            hypothenuseLength = max([e.length() for e in self.edges])
            shortSides = []
            for edge in self.edges:
                if edge.length() < hypothenuseLength:
                    # edge is not the hypothenuse
                    shortSides.append(edge.vector())
            self.determinant = abs(shortSides[0][0] * shortSides[1][1] - shortSides[0][1] * shortSides[1][0])
        return self.determinant

    def __str__(self):
        out = ["TRIANGLE:"]
        # nodes
        out.append("-----> NODES:")
        for i, node in enumerate(self.nodes):
            out.append(f"\t {i+1}. {str(node)}")

        # dofs
        out.append("--> DOFS:")
        for i, node in enumerate(self.dofs):
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
    def __init__(self, nodes, edges, triangles, degree = 1):
        self.nodes     = nodes
        self.edges     = edges
        self.triangles = triangles
        self.degree    = degree
        self.dofs      = self.nodes[:]
        self.distributeDofs()

    def plot(self, title = 'Finite Element Grid'):
        showLabels = True if len(self.nodes) < 100 else False

        # plot edges
        for i,edge in enumerate(self.edges):
            n1, n2 = edge.start, edge.end
            plt.plot([n1.x,n2.x], [n1.y,n2.y], color = "blue")
            middle = n1.midpoint(n2)
            if showLabels:
                plt.text(middle.x, middle.y, str(i+1), color = "blue")

        # plot nodes
        xCoords, yCoords = [], []
        for i,node in enumerate(self.nodes):
            xCoords.append(node.x)
            yCoords.append(node.y)
            if showLabels:
                plt.text(node.x,node.y, str(i+1))
        plt.scatter(xCoords, yCoords, color = "black")

        # label the triangles
        if showLabels:
            for i, triangle in enumerate(self.triangles):
                n1, n2, n3 = triangle.nodes
                middle = n1.midpoint(n2.midpoint(n3))
                plt.text(middle.x, middle.y, str(i+1), color = "red")

        plt.title(title)
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

        # dofs
        out.append("--> DOFS:")
        for i, node in enumerate(self.dofs):
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

    def refine(self):
        refinedNodes     = self.nodes[:]
        refinedEdges     = []
        refinedTriangles = []

        for edge in self.edges:
            childrenEdges, midpoint = edge.getChildren()
            midpoint.ind = len(refinedNodes)
            refinedNodes.append(midpoint)
            refinedEdges += childrenEdges

        for triangle in self.triangles:
            # prepare datastructures for centerTriangle
            centerNodes = []
            centerEdges = []
            for i, node in enumerate(triangle.nodes):
                # compute the new triangle
                newTriangleEdges    = []
                newTriangleNodes    = [node]
                newTriangleMaterial = triangle.material
                # add two existing nodes and two existing edges
                for j, edge in enumerate(triangle.edges):
                    if i != j:
                        childrenEdges, midpoint = edge.getChildren()
                        newTriangleNodes.append(midpoint)
                        centerNodes.append(midpoint)
                        for childEdge in childrenEdges:
                            if node in childEdge.nodes():
                                newTriangleEdges.append(childEdge)
                # compute remaining edge
                newEdge = Edge(newTriangleNodes[1], newTriangleNodes[2])
                newTriangleEdges.append(newEdge)
                refinedEdges.append(newEdge)
                centerEdges.append(newEdge)

                refinedTriangles.append(
                    Triangle(newTriangleNodes, newTriangleEdges, newTriangleMaterial)
                )
            # add center triangle
            refinedTriangles.append(
                Triangle(list(set(centerNodes)), centerEdges, triangle.material)
            )

        return Grid(refinedNodes,refinedEdges,refinedTriangles, degree = self.degree)

    def distributeDofs(self):
        if self.degree == 3:
            raise NotImplementedError()

        if self.degree == 2:
            for edge in self.edges:
                _, midpoint = edge.getChildren()
                midpoint.ind = len(self.dofs)
                self.dofs.append(midpoint)

        for triangle in self.triangles:
            triangle.distributeDofs(degree = self.degree)

def homeworkGrid(degree = 1):
    # create nodes of L-shape
    nodes = []
    for x in [-1.,0.,1.]:
        for y in [-1.,0.,1.]:
            if x != 1. or y != 1.:
                nodes.append(Node(x,y, len(nodes)))

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

    return Grid(nodes, edges, triangles, degree = degree)
