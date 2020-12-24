# 18110053_NguyenQuocBao_lab5_AI
from collections import defaultdict
from queue import PriorityQueue


def min_cost(cost_a, cost_b):
    if (cost_a < cost_b):
        return cost_a
    return cost_b

class Graph :

    def __init__(self,vertices,name_ct, heuristic,matrix):
        self.vertices = vertices
        self.name_ct = name_ct
        self.heuristic = heuristic
        self.graph = defaultdict(list)
        self.matrix = matrix
        
    
    # add edge for graph
    def add_edge(self,src,dest): 
        self.graph[src].append([dest,self.name_ct[dest],self.heuristic[dest],self.matrix[src][dest]]) 

    # Show infomation graph
    def show_graph(self):
        for i in range(self.vertices):
            # print(self.graph[i].items())
            print((i, self.name_ct[i]),  ':', self.graph[i])

    # function sort distance between vertices 
    def sort_weight(self):
        custom = [] # Array stores [current address, destination address, distance]
        for i in range(len(self.graph)):
            for Node in self.graph[i]:
                visited,Cost=Node[0],Node[-1]
                custom.append([i,visited,Cost])
        return sorted(custom ,key=lambda item: item[2])

    # A* search
    def aStart(self,start,goal):
        Open = PriorityQueue()
        f = 0; g = 0; h = 0
        root = (f,g,h,start)
        Open.put(root)
        Close = []
        father = {}
        path = [goal]
        while True:
            if Open.empty():
                raise Exception("No way Exception")
            current_f,current_h,current_g,current_node = Open.get()
            Close.append(current_node)
            if current_node == goal:
                key = goal
                while key in father.keys():
                    value = father.pop(key)
                    path.append(value)
                    key = value
                    if key == start:
                        break
                path.reverse()
                return path
            if current_node not in self.graph:
                continue
            for Node in self.graph[current_node]:
                node,grid,heuris = Node[0],Node[-2],Node[-1]
                grid = current_g + min_cost(current_g,grid)
                f = grid + heuris
                if node not in Close:
                    Open.put((f, grid, heuris, node))
                    if node not in father.keys():
                        father[node] = current_node
                        
    # A utility function to find set of an element i
    # (uses path compression technique)
    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])
 
    # A function that does union of two sets of x and y
    # (uses union by rank)
    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)
 
        # Attach smaller rank tree under root of
        # high rank tree (Union by Rank)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
 
        # If ranks are same, then make one as root
        # and increment its rank by one
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    def Kruskal_algo(self):
        result = [] # This will store the resultant MST
        Sorted = self.sort_weight()
        parent = []
        rank =[]
        # An index variable, used for sorted edges
        i = 0
        # An index variable, used for result[]
        e = 0
        for node in range(self.vertices):
            parent.append(node)
            rank.append(0)
        while e < self.vertices - 1:
 
            # Step 2: Pick the smallest edge and increment
            # the index for next iteration
            node, visited, cost = Sorted[i]
            i = i + 1
            x = self.find(parent, node)
            y = self.find(parent, visited)
 
            # If including this edge does't
            #  cause cycle, include it in result 
            #  and increment the indexof result 
            # for next edge
            if x != y:
                e = e + 1
                result.append([node, visited, cost])
                self.union(parent, rank, x, y)
            # Else discard the edge
 
        minimumCost = 0
        print("Edges in the constructed MST")
        for node, visited, cost in result:
            minimumCost += cost
            print("{} -- {} = {}".format(self.name_ct[node],self.name_ct[visited], cost))    
        print("Minimum Spanning Tree" , minimumCost)


def read_txt(file):
    vertices = int(file.readline())
    start, goal = [int(num) for num in file.readline().split(' ')]
    name_ct = [item for item in file.readline()[:-1].split('\t')]
    heuristic = [int(num) for num in file.readline().split('\t')]
    matrix = [[int(num) for num in line.split('\t')] for line in file]
    return vertices,start,goal,name_ct,heuristic,matrix



def convert_graph(vertices,name_ct,heuristic,matrix):
    graph = Graph(vertices,name_ct,heuristic,matrix)
    for i in range(vertices):
        for j in range(vertices):
            if (matrix[i][j] != 0):
                graph.add_edge(i,j)
    return graph


        

def main():
    file_1 = open("Input.txt","r")
    vertices, start,goal,name_ct,heuristic,matrix = read_txt(file_1)
    # print('Vertices: ',vertices,'\n','Start: ' ,start,'Goal: ' ,goal,'\n')
    graph = convert_graph(vertices,name_ct,heuristic,matrix)
    # graph.show_graph()
    print('\n')
    graph.Kruskal_algo()
    print('\n')
    print('path A*:')
    path_aStart = graph.aStart(9,1)
    print(" --> ".join(name_ct[i] for i in path_aStart))

    


if __name__ == "__main__":
    main()