# Nguyễn Quốc Bảo
# 18110053
# Bài tập thực hành tuần 1


#*******************************

def BFS(graph, start, goal):
    # keep track of explored nodes
    explored = []
    # keep track of all the paths to be checked
    queue = [[start]]
 
    # return path if start is goal
    if start == goal:
        return "Start = goal"
 
    while queue:
        # pop the first path from the queue
        path = queue.pop(0)
        # get the last node from the path
        node = path[-1] 
        if node not in explored:
            if node not in graph:
                continue
            neighbours = graph[node]
            # go through all neighbour nodes, construct a new path and
            # push it into the queue
            for neighbour in neighbours:
                new_path = list(path)
                new_path.append(neighbour)
                queue.append(new_path)
                # return path if neighbour is goal
                if neighbour == goal:
                    return new_path
 
            # mark node as explored
            explored.append(node)
 
    # in case there's no path between the 2 nodes
    return "There's no path between the 2 nodes"


def DFS(graph, start, goal):
    # keep track of explored nodes
    explored = []
    # keep track of all the paths to be checked
    queue = [[start]]
 
    # return path if start is goal
    if start == goal:
        return "Start = goal"
 
    while queue:
        # pop the first path from the queue
        path = queue.pop()
        # get the last node from the path
        node = path[-1]
        if node not in explored:
            if node not in graph:
                continue
            neighbours = graph[node]
            # go through all neighbour nodes, construct a new path and
            # push it into the queue
            for neighbour in neighbours:
                new_path = list(path)
                new_path.append(neighbour)
                queue.append(new_path)
                # return path if neighbour is goal
                if neighbour == goal:
                    return new_path
 
            # mark node as explored
            explored.append(node)
 
    # in case there's no path between the 2 nodes
    return "There's no path between the 2 nodes"


    
def UCS(graph, start, goal):
    # keep track of explored nodes
    explored = []
    # keep track of all the paths to be checked
    queue = [([start],0)]
    # return path if start is goal
    if start == goal:
        return "Start = goal"

    while queue:
        # pop the first path from the queue
        queue.sort(key = lambda tup: tup[1])
        path,current_cost = queue.pop(0)
        # get the last node from the path
        current_node = path[-1]
        if current_node == goal:
            return current_cost,path
        if current_node not in explored:
            if current_node not in graph:
                continue
            neighbours = graph[current_node]
            # go through all neighbour nodes, construct a new path and
            # push it into the queue
            for neighbour in neighbours:
                node, cost = neighbour
                new_path = list(path)
                new_path.append(node)
                queue.append((new_path,current_cost+cost))
                # return path if neighbour is goal
                # if node == goal:
                #     return current_cost+cost,new_path
 
            # mark node as explored
            explored.append(current_node)
 
    # in case there's no path between the 2 nodes
    return None,"There's no path between the 2 nodes"

def read_txt(file):
    size = int(file.readline())
    start, goal = [int(num) for num in file.readline().split(' ')]
    matrix = [[int(num) for num in line.split(' ')] for line in file]
    return size,start,goal,matrix

def convert_graph(vertices,matrix):
    dicts = {}
    for i in range(vertices):
        ls = []
        for j in range(vertices):
            if (matrix[i][j] != 0):
                ls.append(j)
        dicts[i] = ls
    return dicts

def convert_graph_weight(vertices,matrix):
    dicts = {}
    for i in range(vertices):
        ls = []
        for j in range(vertices):
            if (matrix[i][j] != 0):
                ls.append((j,matrix[i][j]))
        dicts[i] = ls
    return dicts

def main():
    file_1 = open("Inp.txt","r")
    file_2 = open("inputUCS.txt","r")
    size_1, start_1,goal_1,matrix_1 = read_txt(file_1)
    size_2, start_2,goal_2,matrix_2 = read_txt(file_2)
    file_1.close()
    file_2.close()
    graph_1 = convert_graph(size_1,matrix_1)
    graph_2 = convert_graph_weight(size_2,matrix_2)
    # print("size, start, goal: ", size_1, start_1, goal_1)
    # result_bfs = BFS(graph_1,start_1,goal_1)
    # print("result of BFS: \n",result_bfs)  
    # result_dfs = DFS(graph_1,start_1,goal_1)
    # print("result of DFS: \n",result_dfs)
    # print("size, start, goal: ", size_2, start_2, goal_2)
    # cost, result_ucs = UCS(graph_2,start_2,goal_2)
    # print("result of UCS: \n",cost, result_ucs)
    # #******************************************************
    # print("\n")
    # print("Set size, start, goal:")
    # size_1 = 18; size_2 = 18
    # start_1 = 0; goal_1 = 17
    # start_2 = 0; goal_2 = 17
    # graph_1 = convert_graph(size_1,matrix_1)
    # graph_2 = convert_graph_weight(size_2,matrix_2)
    # print("size, start, goal: ", size_1, start_1, goal_1)
    # result_bfs = BFS(graph_1,start_1,goal_1)
    # print("result of BFS: \n",result_bfs)  
    # result_dfs = DFS(graph_1,start_1,goal_1)
    # print("result of DFS: \n",result_dfs)
    # print("size, start, goal: ", size_2, start_2, goal_2)
    # cost, result_ucs = UCS(graph_2,start_2,goal_2)
    # print("result of UCS: \n",cost, result_ucs)

    for i in graph_1:
        print(i, graph_1[i])

    # print(graph_1)
    
if __name__ == "__main__":
    main()    

