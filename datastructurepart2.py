import collections
import heapq

"""
Implementation of topological sort, find shortest paths in as DAG, Dijsktra 
algorithm and Bellman-Ford algorithm. For this examples I used name tuple 
as the edge definition instead of using the class Edge. 
The testing examples were taking of the course Algorithms II from Princeton
(coursera.org).
"""

Edge = collections.namedtuple('Edge', 'from_,to,weight')


class EdgeWeightedDigraph:
    def __init__(self, vertices):
        # Number of vertices
        self.__vertices = vertices
        self.__edges = 0
        self.__adjacent = {}
        for i in range(vertices):
            self.__adjacent[i] = set()

    def addEdge(self, e):
        v = e.from_
        self.__adjacent[v].add(e)
        self.__edges += 1

    def number_vertices(self):
        return self.__vertices

    def vertexNeighbors(self, vertex):
        if vertex < self.__vertices:
            return self.__adjacent[vertex]
        else:
            return None

    def edges(self):
        """
        A difference with EdgeWeightedGraph() is that we don't have to 
        validate if the edge is already in the list because this is a
        digraph
        """
        edgeslist = []
        for v in range(self.__vertices):
            for edge in self.__adjacent[v]:
                edgeslist.append(edge)
        return edgeslist


class TopologicalSort:
    def __init__(self, digraph):
        self.graph = digraph
        self.stack = []
        self.vertices = digraph.number_vertices()
        self.visited = [False]*self.graph.number_vertices()

    def __depthfirstpathshelper(self, vertex):
        self.visited[vertex] = True
        verticesTovisit = self.graph.vertexNeighbors(vertex)
        for from_, vertexTovisit, weight in verticesTovisit:
            if not(self.visited[vertexTovisit]):
                self.__depthfirstpathshelper(vertexTovisit)
        self.stack.append(vertex)

    def topologicalsort(self):
        for i in range(self.graph.number_vertices()):
            if not(self.visited[i]):
                self.__depthfirstpathshelper(i)
        return self.stack[::-1]


# test topological sort ignoring the weights
testdiagrap = EdgeWeightedDigraph(7)
testdiagrap.addEdge(Edge(0, 5, 1))
testdiagrap.addEdge(Edge(0, 1, 1))
testdiagrap.addEdge(Edge(3, 5, 1))
testdiagrap.addEdge(Edge(5, 2, 1))
testdiagrap.addEdge(Edge(6, 0, 1))
testdiagrap.addEdge(Edge(1, 4, 1))
testdiagrap.addEdge(Edge(0, 2, 1))
testdiagrap.addEdge(Edge(3, 6, 1))
testdiagrap.addEdge(Edge(3, 4, 1))
testdiagrap.addEdge(Edge(6, 4, 1))
testdiagrap.addEdge(Edge(3, 2, 1))
testtopsort = TopologicalSort(testdiagrap)
result = testtopsort.topologicalsort()
print(result)


class SPDAG:
    """
    Find shortest path for DAG. Uses the topological sort class
    """

    def __init__(self, graph):
        self.digraph = graph
        self.distTo = [float('inf')]*self.digraph.number_vertices()
        self.edgeTo = [None]*self.digraph.number_vertices()

    def __relax(self, edge):
        if self.distTo[edge.to] > self.distTo[edge.from_] + edge.weight:
            self.distTo[edge.to] = self.distTo[edge.from_] + edge.weight
            self.edgeTo[edge.to] = edge

    def shortpaths(self):
        # Step 1: topological sort
        topsortorder = TopologicalSort(self.digraph)
        order = topsortorder.topologicalsort()
        self.distTo[order[0]] = 0.0
        for vertex in order:
            edges = self.digraph.vertexNeighbors(vertex)
            for edge in edges:
                self.__relax(edge)


# Testing
testdag = EdgeWeightedDigraph(8)
testdag.addEdge(Edge(0, 1, 5.0))
testdag.addEdge(Edge(0, 4, 9.0))
testdag.addEdge(Edge(0, 7, 8.0))
testdag.addEdge(Edge(1, 2, 12.0))
testdag.addEdge(Edge(1, 3, 15.0))
testdag.addEdge(Edge(1, 7, 4.0))
testdag.addEdge(Edge(2, 3, 3.0))
testdag.addEdge(Edge(2, 6, 11.0))
testdag.addEdge(Edge(3, 6, 9.0))
testdag.addEdge(Edge(4, 5, 4.0))
testdag.addEdge(Edge(4, 6, 20.0))
testdag.addEdge(Edge(4, 7, 5.0))
testdag.addEdge(Edge(5, 2, 1.0))
testdag.addEdge(Edge(5, 6, 13.0))
testdag.addEdge(Edge(7, 5, 6.0))
testdag.addEdge(Edge(7, 2, 7.0))
testtopsort = TopologicalSort(testdag)
#result = testtopsort.topologicalsort()
# print(result)
shortestpaths = SPDAG(testdag)
shortestpaths.shortpaths()
print(shortestpaths.distTo)


class Dijkstraqueue:
    def __init__(self, key, dist):
        self.key = key
        self.dist = dist

    def __repr__(self):
        return f'queue vertex: {self.key}'

    def __lt__(self, other):
        return self.dist < other.dist

    def __eq__(self, other):
        return self.key == other.key

# Dijkstra


class Dijkstra:
    def __init__(self, graph):
        self.digraph = graph
        self.distTo = [float('inf')]*self.digraph.number_vertices()
        self.edgeTo = [None]*self.digraph.number_vertices()
        self.pq = []
        heapq.heapify(self.pq)

    def __relax(self, edge):
        if self.distTo[edge.to] > self.distTo[edge.from_] + edge.weight:
            prevdist = self.distTo[edge.to]
            self.distTo[edge.to] = self.distTo[edge.from_] + edge.weight
            self.edgeTo[edge.to] = edge
            if Dijkstraqueue(edge.to, prevdist) in self.pq:
                self.pq.remove(Dijkstraqueue(edge.to, prevdist))

            heapq.heappush(self.pq, Dijkstraqueue(
                edge.to, self.distTo[edge.to]))

    def findshortpath(self, initvertex):
        self.distTo[initvertex] = 0.0
        heapq.heappush(self.pq, Dijkstraqueue(
            initvertex, self.distTo[initvertex]))
        while self.pq:
            vertex = heapq.heappop(self.pq)
            edges = self.digraph.vertexNeighbors(vertex.key)
            for edge in edges:
                self.__relax(edge)


# Testing
testdijsktra = EdgeWeightedDigraph(8)
testdijsktra.addEdge(Edge(0, 1, 5.0))
testdijsktra.addEdge(Edge(0, 4, 9.0))
testdijsktra.addEdge(Edge(0, 7, 8.0))
testdijsktra.addEdge(Edge(1, 2, 12.0))
testdijsktra.addEdge(Edge(1, 3, 15.0))
testdijsktra.addEdge(Edge(1, 7, 4.0))
testdijsktra.addEdge(Edge(2, 3, 3.0))
testdijsktra.addEdge(Edge(2, 6, 11.0))
testdijsktra.addEdge(Edge(3, 6, 9.0))
testdijsktra.addEdge(Edge(4, 5, 4.0))
testdijsktra.addEdge(Edge(4, 6, 20.0))
testdijsktra.addEdge(Edge(4, 7, 5.0))
testdijsktra.addEdge(Edge(5, 2, 1.0))
testdijsktra.addEdge(Edge(5, 6, 13.0))
testdijsktra.addEdge(Edge(7, 5, 6.0))
testdijsktra.addEdge(Edge(7, 2, 7.0))
dijkstratest = Dijkstra(testdijsktra)
dijkstratest.findshortpath(0)
print('Dijkstra from 0')
print(dijkstratest.distTo)
print('Dijkstra from 1')
dijkstratest2 = Dijkstra(testdijsktra)
dijkstratest2.findshortpath(1)
print(dijkstratest2.distTo)

# Bellman-ford


class Bellman_Ford:
    def __init__(self, graph):
        self.digraph = graph
        self.distTo = [float('inf')]*self.digraph.number_vertices()
        self.edgeTo = [None]*self.digraph.number_vertices()
        self.containNegativecycle = False

    def __relax(self, edge):
        if self.distTo[edge.to] > self.distTo[edge.from_] + edge.weight:
            self.distTo[edge.to] = self.distTo[edge.from_] + edge.weight
            self.edgeTo[edge.to] = edge

    def bfshortestpaths(self, initvertex):
        self.distTo[initvertex] = 0.0
        for _ in range(1, self.digraph.number_vertices()):
            for edge in self.digraph.edges():
                self.__relax(edge)
        for edge in self.digraph.edges():
            if self.distTo[edge.to] > self.distTo[edge.from_] + edge.weight:
                self.containNegativecycle = True
                break
        return self.containNegativecycle


# testing bellman-ford
testbellman = EdgeWeightedDigraph(8)
testbellman.addEdge(Edge(0, 1, 5.0))
testbellman.addEdge(Edge(0, 4, 9.0))
testbellman.addEdge(Edge(0, 7, 8.0))
testbellman.addEdge(Edge(1, 2, 12.0))
testbellman.addEdge(Edge(1, 3, 15.0))
testbellman.addEdge(Edge(1, 7, 4.0))
testbellman.addEdge(Edge(2, 3, 3.0))
testbellman.addEdge(Edge(2, 6, 11.0))
testbellman.addEdge(Edge(3, 6, 9.0))
testbellman.addEdge(Edge(4, 5, 4.0))
testbellman.addEdge(Edge(4, 6, 20.0))
testbellman.addEdge(Edge(4, 7, 5.0))
testbellman.addEdge(Edge(5, 2, 1.0))
testbellman.addEdge(Edge(5, 6, 13.0))
testbellman.addEdge(Edge(7, 5, 6.0))
testbellman.addEdge(Edge(7, 2, 7.0))
bellman_ford = Bellman_Ford(testbellman)
hasnegativecycle = bellman_ford.bfshortestpaths(0)
print('Bellman-ford')
print(f'has negative cycle {hasnegativecycle}')
print(bellman_ford.distTo)
# negative cycle test
testbellman2 = EdgeWeightedDigraph(8)
testbellman2.addEdge(Edge(4, 5, 0.35))
testbellman2.addEdge(Edge(5, 4, -0.66))
testbellman2.addEdge(Edge(4, 7, 0.37))
testbellman2.addEdge(Edge(5, 7, 0.28))
testbellman2.addEdge(Edge(7, 5, 0.28))
testbellman2.addEdge(Edge(5, 1, 0.32))
testbellman2.addEdge(Edge(0, 4, 0.38))
testbellman2.addEdge(Edge(0, 2, 0.26))
testbellman2.addEdge(Edge(7, 3, 0.39))
testbellman2.addEdge(Edge(1, 3, 0.29))
testbellman2.addEdge(Edge(2, 7, 0.34))
testbellman2.addEdge(Edge(6, 2, 0.40))
testbellman2.addEdge(Edge(3, 6, 0.52))
testbellman2.addEdge(Edge(6, 0, 0.58))
testbellman2.addEdge(Edge(6, 4, 0.93))
bellman_ford2 = Bellman_Ford(testbellman2)
hasnegativecycle = bellman_ford2.bfshortestpaths(0)
print('Bellman-ford - Negative cycle testing')
print(f'has negative cycle {hasnegativecycle}')
print(bellman_ford.distTo)
