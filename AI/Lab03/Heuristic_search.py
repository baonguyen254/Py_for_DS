# 18110053_NguyenQuocBao_lab3_AI
from queue import PriorityQueue
from collections import defaultdict


def min_cost(cost_a, cost_b):
    if (cost_a < cost_b):
        return cost_a
    return cost_b


class Graph :

    def __init__(self,vertices,name_ct,heuristic,matrix):
        self.heuristic = heuristic
        self.name_ct = name_ct
        self.graph = defaultdict(list)
        self.matrix = matrix
        self.vertices = vertices
    
    # add edge for graph
    def add_edge(self,src,dest): 
        self.graph[src].append((dest,self.name_ct[dest],self.matrix[src][dest],self.heuristic[dest])) 

    # Show infomation graph
    def show_graph(self):
        for i in range(self.vertices):
            print((i, self.name_ct[i]),  ':', self.graph[i])

    # Geedy best first search 
    def GBFS(self,start,goal):
        frontier = PriorityQueue()
        root = (start,self.heuristic[start])
        frontier.put(root)
        explored = []
        father = {}
        path = [goal]
        while True:
            if frontier.empty():
                raise Exception("No way Exception")
            current_node, current_h = frontier.get()
            explored.append(current_node)
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
                node ,heuris = Node[0], Node[-1]
                if node not in explored:
                    frontier.put((node, heuris))
                    if node not in father.keys():
                        father[node] = current_node
    
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


def read_txt(file):
    vertices = int(file.readline())
    start, goal = [int(num) for num in file.readline().split(' ')]
    name_ct = [item for item in file.readline()[:-1].split('\t')]
    heuristic = [int(num) for num in file.readline().split('\t')]
    # heuristic = [(heuristic[i],weight[i]) for i in range(size)]
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
    print('Vertices: ',vertices,'\n','Start: ' ,start,'Goal: ' ,goal,'\n')
    graph = convert_graph(vertices,name_ct,heuristic,matrix)
    # graph.show_graph()
    print('\n')
    print('path GBFS:')
    path_GBFS = graph.GBFS(0,7)
    print(" --> ".join(name_ct[i] for i in path_GBFS))
    print('\n')
    print('path A*:')
    path_aStart = graph.aStart(0,7)
    print(" --> ".join(name_ct[i] for i in path_aStart))


if __name__ == "__main__":
    main()