from math import sqrt
from collections import defaultdict
import pyvisgraph as vg
import queue
from queue import PriorityQueue

def read_file(file):
    total_polygons,x1,y1,x2,y2 = [float(num) for num in file.readline()[:-1].split(',')]
    total_polygons = int(total_polygons)
    start = (x1,y1, 'S')
    goal = (x2,y2, 'G')
    data =  file.readlines()
    polygons = {}
    for k in range(total_polygons):
        list_point = []
        for i in range(0, len(data)):
            if ',' not in (data[i].rstrip()):
                vertices = []
                num_of_point = int(data[i].rstrip())
                for item in data[i+1:i+num_of_point+1]:
                    value = [float(j) for j in item.rstrip().split(',')]
                    vertices.append((value[0],value[1],k))
                list_point.append(vertices)    
            else:    
                continue         
        polygons[k] = list_point[k]
    return total_polygons, start, goal, polygons

# calculation distance with other point
def distance(point_1, point_2):
    """ calculation distance with 2 point """
    dx = point_1[0] - point_2[0]
    dy = point_1[1] - point_2[1]
    return sqrt(dx**2 + dy**2)


# Check the relative position of the point
def Is_same_side(edge, point_1, point_2):
    ''' Check the relative position of the point'''
    x1 = edge[0][0]
    y1 = edge[0][1]
    x2 = edge[1][0]
    y2 = edge[1][1]
    d1 = (point_1[0] - x1)*(y2 - y1) - (point_1[1] - y1)*(x2 - x1)
    d2 = (point_2[0] - x1)*(y2 - y1) - (point_2[1] - y1)*(x2 - x1)
    return d1*d2 < 0

# Check for the adjacent points of 1 point
def adjoining_points(total_polygons,polygons):
    ''' Return the adjacent points of each point in the polygon'''
    adjacent_points = defaultdict(list)
    edges = all_edges(total_polygons,polygons)
    for edge in edges:
        point1 = edge[0]
        point2 = edge[1]
        adjacent_points[point1].append(point2)
        adjacent_points[point2].append(point1)
    return adjacent_points

def successor(all_points, all_edges, adjacent_points):
    '''get all point can see and can't see of each point'''
    graph = defaultdict(list)
    not_see = defaultdict(list)
    for current_point in all_points:
        for edge in all_edges:
            for point in all_points:
                # check point and current point the same
                if (point == current_point) | (point in edge):
                    continue
                # check point and current point have the same polygon and edge
                if ((point[2] == current_point[2]) \
                    and (point not in adjacent_points[current_point])):
                    if point not in not_see[current_point]:
                        not_see[current_point].append(point)
                else:
                    edge1 = (current_point, edge[0])
                    edge2 = (current_point, edge[1])
                    # Check the relative position of the point
                    if (Is_same_side(edge1, edge[1], point)==False \
                        and Is_same_side(edge2, edge[0], point)==False\
                             and Is_same_side(edge, current_point, point)==True):
                        if point in graph[current_point]:
                            graph[current_point].remove(point)
                        if point not in not_see[current_point]:
                            not_see[current_point].append(point)
                    else:
                        if point not in not_see[current_point]:
                            if point not in graph[current_point]:    
                                graph[current_point].append(point)
    return graph

def all_edges(total_polygons, polygons):
    ''' get all edge of each polygon'''
    edges = []
    for i in range(total_polygons):
        pol = polygons[i]
        num_ver = len(polygons[i])
        for j in range(num_ver):
            edge = [pol[j % num_ver], pol[(j + 1) % num_ver]]
            edges.append(edge)
    return edges

def all_points(start,goal,polygons):
    ''' get all point of all polygon '''
    all_point = [start,goal]
    for i in range(len(polygons)):
        for point in polygons[i]:
            all_point.append(point)
    return all_point



def shortest_path(graph, start, goal):
    ''' Return path shortest from start to goal'''
    explored = []
    queue = PriorityQueue()
    root = (0,[start])
    queue.put(root)
    if (start == goal):
        return "Start = goal"
    while True:
        if queue.empty():
            raise Exception("No way Exception")
        current_distance, path = queue.get()
        current_point = path[-1]
        if current_point == goal:
            print('total {} path'.format(queue.qsize()))
            return current_distance,path
        if current_point not in explored:
            if current_point not in graph:
                continue
            list_point = graph[current_point]
            # go through all point can see, construct a new path and
            # push it into the queue
            for point in list_point:
                d = distance(current_point,point)
                new_path = list(path)
                new_path.append(point)
                queue.put((current_distance+d,new_path))
 
            # mark point as explored
            explored.append(current_point)
    # in case there's no path between the 2 points
    return None,"There's no path between the 2 points"
    
        


def main():
    # Preprocessing
    file = open("input.txt","r")
    total_polygons, start, goal, polygons = read_file(file)
    all_edge = all_edges(total_polygons,polygons)
    all_point = all_points(start,goal,polygons)
    adjoining = adjoining_points(total_polygons,polygons)
    print('\n\n')
    print('start:',start)
    print('goal',goal)
    print('total polygon: ', total_polygons)
    # Set of point can see
    graph = successor(all_point,all_edge,adjoining)
    # Search shortest path and distance 
    distance, path = shortest_path(graph,start,goal)
    print('the shortest path from {} to {}:'.format(start,goal))
    print(" --> ".join(str(i) for i in path))
    print('Distance: ', distance)
    print('\n\n')
    

   
if __name__ == "__main__":
    main()