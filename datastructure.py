import operator
from collections import deque, namedtuple
import heapq

# binary heap


class BinaryHeap():
    def __init__(self):
        self.array = []
        self.n = 0

    def swim(self, k):
        j = k//2
        while k > 1 and self.array[j] < self.array[k]:
            temp = self.array[j]
            self.array[j] = self.array[k]
            self.array[k] = temp
            j = k//2

    def insert(self, val):
        if len(self.array) == 0:
            self.array.append('-')
        self.array.append(val)
        self.n += 1
        self.swim(self.n)

    def sink(self, k):
        while 2*k <= self.n:
            j = 2*k
            if j < self.n and self.array[j] < self.array[j+1]:
                j += 1
            if self.array[k] >= self.array[j]:
                break
            temp = self.array[j]
            self.array[j] = self.array[k]
            self.array[k] = temp
            k = j

    def delMax(self):
        val = self.array[1]
        self.array[1] = self.array[self.n]
        self.array[self.n] = val
        self.array.pop()
        self.sink(1)
        self.n -= 1
        return val


"""
Create the simplest implementation where every Vertex is a key from 0
to the number of vertex - 1. Based on Algorithms II coursera.org 
(undirected graphs)
"""


class Graphs():
    def __init__(self, vertices):
        # Number of vertices
        self.__vertices = vertices
        self.__adjacent = {}
        for i in range(vertices):
            self.__adjacent[i] = []

    def addEdge(self, vertexv, vertexw):
        self.__adjacent[vertexv].append(vertexw)
        self.__adjacent[vertexw].append(vertexv)

    def number_vertices(self):
        return self.__vertices

    def vertexNeighbors(self, vertex):
        if vertex < self.__vertices:
            return self.__adjacent[vertex]
        else:
            return None


class DepthFirstPaths():
    def __init__(self, graph):
        vertices = graph.number_vertices()
        self.visited = [False]*vertices
        self.edgeTo = [None]*vertices

    def __depthfirstpathshelper(self, graph, vertex):
        self.visited[vertex] = True
        verticesTovisit = graph.vertexNeighbors(vertex)
        for vertexTovisit in verticesTovisit:
            if operator.not_(self.visited[vertexTovisit]):
                self.__depthfirstpathshelper(graph, vertexTovisit)
                self.edgeTo[vertexTovisit] = vertex

    def depthfirstpaths(self, graph, initvertex):
        self.__depthfirstpathshelper(graph, initvertex)


class BreathFistSearch():
    def __init__(self, graph):
        vertices = graph.number_vertices()
        self.visited = [False]*vertices
        self.edgeTo = [None]*vertices

    def breathfirstsearch(self, graph, initvertex):
        # create a queue
        toVisit = deque()
        toVisit.append(initvertex)
        self.visited[initvertex] = True
        while toVisit:
            currentvertex = toVisit.popleft()
            adjcurrentvertex = graph.vertexNeighbors(currentvertex)
            for vertex in adjcurrentvertex:
                if operator.not_(self.visited[vertex]):
                    # enqueue
                    toVisit.append(vertex)
                    self.visited[vertex] = True
                    self.edgeTo[vertex] = currentvertex


# Test
# testGraph = Graphs(6)
# testGraph.addEdge(0, 1)
# testGraph.addEdge(0, 2)
# testGraph.addEdge(0, 5)
# testGraph.addEdge(1, 2)
# testGraph.addEdge(2, 3)
# testGraph.addEdge(2, 4)
# testGraph.addEdge(3, 4)
# testGraph.addEdge(3, 5)

# print(testGraph.vertexNeighbors(0))
# testDFS = DepthFirstPaths(testGraph)
# testDFS.depthfirstpaths(testGraph, 0)
# print(testDFS.edgeTo)
# testBFS = BreathFistSearch(testGraph)
# testBFS.breathfirstsearch(testGraph, 0)
# print(testBFS.edgeTo)


"""
Greedy algorithms to find MST. Based on Algorithms II coursera.org 
(undirected graphs)
"""


class Edge:
    def __init__(self, v, w, weight):
        self.__v = v
        self.__w = w
        self.weight = weight

    def bringOther(self, other):
        if other == self.__v:
            return self.__w
        elif other == self.__w:
            return self.__v
        else:
            return ValueError('Inconsistent edge')

    def bringEither(self):
        return self.__v

    def __lt__(self, nxt):
        return self.weight < nxt.weight


class EdgeWeightedGraph():
    def __init__(self, vertices):
        # Number of vertices
        self.__vertices = vertices
        self.__edges = 0
        self.__adjacent = {}
        for i in range(vertices):
            self.__adjacent[i] = []

    def addEdge(self, e):
        v = e.bringEither()
        w = e.bringOther(v)
        self.__adjacent[v].append(e)
        self.__adjacent[w].append(e)
        self.__edges += 1

    def number_vertices(self):
        return self.__vertices

    def vertexNeighbors(self, vertex):
        if vertex < self.__vertices:
            return self.__adjacent[vertex]
        else:
            return None

    def edges(self):
        edgeslist = []
        for v in range(self.__vertices):
            for edge in self.__adjacent[v]:
                if edge.bringOther(v) > v:
                    edgeslist.append(edge)
        return edgeslist


class QuickFinds:

    def __init__(self, val):
        self.array = [i for i in range(0, val)]

    def connected(self, p, q):
        return self.array[p] == self.array[q]

    def union(self, p, q):
        val_p = self.array[p]
        #self.array[p] = self.array[q]
        for _ in range(self.array.count(val_p)):
            self.array[self.array.index(val_p)] = self.array[q]


class KruskalMST:
    def __init__(self):
        self.MST = []

    def kruskalmst(self, edgeweihtedgralp):
        quickfind = QuickFinds(edgeweihtedgralp.number_vertices())
        edges = edgeweihtedgralp.edges()
        heapq.heapify(edges)
        while edges:
            edge = heapq.heappop(edges)
            v = edge.bringEither()
            w = edge.bringOther(v)
            if not(quickfind.connected(v, w)):
                self.MST.append(edge)
                quickfind.union(v, w)


class LazyPrimMST:
    def __init__(self, vertices):
        self.MST = []
        self.vertMST = [False]*vertices
        self.edges = []
        heapq.heapify(self.edges)

    def visit(self, edgeweihtedgralp, v):
        self.vertMST[v] = True
        for edge in edgeweihtedgralp.vertexNeighbors(v):
            heapq.heappush(self.edges, edge)

    def lazyprimmst(self, edgeweihtedgralp):
        self.visit(edgeweihtedgralp, 0)
        vertices = edgeweihtedgralp.number_vertices()
        while self.edges and len(self.MST) < vertices - 1:
            edge = heapq.heappop(self.edges)
            v = edge.bringEither()
            w = edge.bringOther(v)
            if self.vertMST[v] and self.vertMST[w]:
                continue
            self.MST.append(edge)
            if not(self.vertMST[v]):
                self.visit(edgeweihtedgralp, v)
            if not(self.vertMST[w]):
                self.visit(edgeweihtedgralp, w)


testEdgeWeightedGraph = EdgeWeightedGraph(8)
testEdgeWeightedGraph.addEdge(Edge(4, 5, 0.35))
testEdgeWeightedGraph.addEdge(Edge(4, 7, 0.37))
testEdgeWeightedGraph.addEdge(Edge(5, 7, 0.28))
testEdgeWeightedGraph.addEdge(Edge(0, 7, 0.16))
testEdgeWeightedGraph.addEdge(Edge(1, 5, 0.32))
testEdgeWeightedGraph.addEdge(Edge(0, 4, 0.38))
testEdgeWeightedGraph.addEdge(Edge(2, 3, 0.17))
testEdgeWeightedGraph.addEdge(Edge(1, 7, 0.19))
testEdgeWeightedGraph.addEdge(Edge(0, 2, 0.26))
testEdgeWeightedGraph.addEdge(Edge(1, 2, 0.36))
testEdgeWeightedGraph.addEdge(Edge(1, 3, 0.29))
testEdgeWeightedGraph.addEdge(Edge(2, 7, 0.34))
testEdgeWeightedGraph.addEdge(Edge(6, 2, 0.40))
testEdgeWeightedGraph.addEdge(Edge(3, 6, 0.52))
testEdgeWeightedGraph.addEdge(Edge(6, 0, 0.58))
testEdgeWeightedGraph.addEdge(Edge(6, 4, 0.93))
# print(testEdgeWeightedGraph.vertexNeighbors(4))

edges = testEdgeWeightedGraph.edges()
edges.sort(key=lambda x: x.weight)
heapq.heapify(edges)
for edge in edges:
    print(edge.weight)

kruskal = KruskalMST()
kruskal.kruskalmst(testEdgeWeightedGraph)
mst = kruskal.MST
for edge in mst:
    print(edge.weight)

lazyprim = LazyPrimMST(testEdgeWeightedGraph.number_vertices())
lazyprim.lazyprimmst(testEdgeWeightedGraph)
mst = lazyprim.MST
for edge in mst:
    print(edge.weight)
