from django.shortcuts import render
from django.http import HttpResponse
from django import forms
import json
import math

ROW = 20
COL = 42


# Create your views here.
def home(request):
    return render(request, 'main.html')


def listdata(request):
    x = {}
    a = {}
    b = [[i for i in range(j * 40, (j + 1) * 40)] for j in range(18)]
    for i in range(18):
        a[str(i)] = b[i]
    c = {'a': a}
    if request.method == "POST":
        name = request.POST

        for item in name.keys():
            x[item] = name.get(item)

        if len(x) == 1:
            return render(request, 'main.html', c)
        else:
            t = list(x.keys())
            wall = []
            print(t, x)
            root = int(x.get('head'))
            node = int(x.get('end'))
            for i in range(len(t)):
                if t[i][:2] == 'ss':
                    wall.append(int(t[i][3:]))
            # print(root, node, wall)
            am = [[0 for i in range(42)] for j in range(20)]
            for i in range(42):
                if not i > 19:
                    am[i][0] = 1
                    am[i][-1] = 1
                am[0][i] = 1
                am[-1][i] = 1

            # print(am)
            for i in range(len(wall)):
                xx = wall[i] // 40
                y = wall[i] % 40
                am[xx + 1][y + 1] = 1

            graph = [[] for i in range(720)]
            for i in range(1, 19):
                for j in range(1, 41):
                    node1 = ((i - 1) * 40) + (j - 1)
                    if am[i - 1][j] == 0:
                        graph[node1].append((((i - 2) * 40) + (j - 1)))
                    if am[i + 1][j] == 0:
                        graph[node1].append((((i) * 40) + (j - 1)))
                    if am[i][j + 1] == 0:
                        graph[node1].append((((i - 1) * 40) + (j)))
                    if am[i][j - 1] == 0:
                        graph[node1].append((((i - 1) * 40) + (j - 2)))

            if int(x.get('rdo')) == 1:
                js = bfs(root, node, graph, wall)
            elif int(x.get('rdo')) == 2:
                parentdfs = [-1] * 720
                visiteddfs = []
                vis = [0] * 720
                ll = 0
                dfs(vis, root, node, graph, parentdfs, visiteddfs, ll)
                ppy = pathdfs(root, node, parentdfs)
                for i in range(len(visiteddfs)):
                    if visiteddfs[i] == [node]:
                        visiteddfs = visiteddfs[:(i + 1)]
                        break
                js = [root, node, wall, ppy, visiteddfs]
            elif int(x.get('rdo')) == 3:
                graphadjm = [[math.inf for i in range(720)] for j in range(720)]
                for i in range(720):
                    graphadjm[i][i] = 0
                    for j in range(len(graph[i])):
                        graphadjm[i][graph[i][j]] = 1
                ppx = dijkstra(graphadjm, root, node)
                # print(ppx)
                pp = ppx[0]
                visitedD = []
                for i in range(len(ppx[1])):
                    visitedD.append([ppx[1][i]])
                js = [root, node, wall, pp, visitedD]
            elif int(x.get('rdo')) == 4:
                src = [(root // 40) + 1, (root % 40) + 1]
                dest = [(node // 40) + 1, (node % 40) + 1]
                # print(src,dest)
                astar1 = search(am, 1, src, dest)
                vi = []
                # ppp1,ppp2=ppp[0],ppp[1]
                for i in range(len(astar1[1])):
                    vi.append([nodeid(astar1[1][i][0], astar1[1][i][1])])
                js = [root, node, wall, astar1[0], vi]

            filetojs = json.dumps(js)
            xyxy = {'a': a, 'filetojs': filetojs}
            x = {}
            return render(request, 'main.html', xyxy)
    else:
        x = {}
        return render(request, 'main.html', c)


def dfs(vis, root1, node, g, parentdfs, visiteddfs, ll):
    vis[root1] = 1
    for i in range(len(g[root1])):

        if vis[g[root1][i]] == 0:
            parentdfs[g[root1][i]] = root1
            visiteddfs.append([g[root1][i]])

            dfs(vis, g[root1][i], node, g, parentdfs, visiteddfs, ll)


def pathdfs(root, node, parentdfs):
    path = [node]
    i = node
    while (i != root):
        i = parentdfs[i]
        path.append(i)
    return path


def bfs(root, node, graph, wall):
    for xoxo in range(1):
        q = [root]
        it = -1
        parent = [-1] * 720
        visited = []
        ans = 0
        while (it < len(q) - 1 and ans == 0):
            it += 1
            p = q[it]
            temp_visited = []

            for i in range(len(graph[p])):
                if parent[graph[p][i]] == -1 and graph[p][i] != root:
                    temp_visited.append(graph[p][i])
                    q.append(graph[p][i])
                    parent[graph[p][i]] = p
                    if graph[p][i] == node:
                        ans = -1
                        break

            visited.append(temp_visited)

        path = [node]
        i = node
        while (parent[i] != -1):
            i = parent[i]
            path.append(i)
        js = [root, node, wall, path, visited]
        x = {}
    return js


def dijkstra(Graph, _s, _d):
    row = len(Graph)
    col = len(Graph[0])
    dist = [math.inf] * row
    Blackened = [0] * row
    pathlength = [0] * row
    parent = [-1] * row
    dist[_s] = 0
    x = []
    for count in range(row - 1):
        u = MinDistance(dist, Blackened)
        if u == math.inf:
            break
        else:
            Blackened[u] = 1
        for v in range(row):

            if Blackened[v] == 0 and Graph[u][v] and dist[u] + Graph[u][v] < dist[v]:
                t = v
                x.append(t)
                parent[v] = u
                pathlength[v] = pathlength[parent[v]] + 1
                dist[v] = dist[u] + Graph[u][v]
            elif Blackened[v] == 0 and Graph[u][v] and dist[u] + Graph[u][v] == dist[v] and pathlength[u] + 1 < \
                    pathlength[v]:
                parent[v] = u
                # xt.append([u])
                pathlength[v] = pathlength[u] + 1
        if dist[_d] != math.inf:
            path1 = pathdfs(_s, _d, parent)
            return [path1, x]


def MinDistance(dist, Blackened):
    min = math.inf
    for v in range(len(dist)):
        if not Blackened[v] and dist[v] < min:
            min = dist[v]
            Min_index = v
    if min == math.inf:
        return math.inf
    else:
        return Min_index


def nodeid(f, g):
    return (((f - 1) * 40) + (g - 1))


class Node:

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def return_path(current_node, maze, vi):
    path = []
    result = []
    current = current_node
    while current is not None:
        path.append(current.position)
        current = current.parent
    path = path[::-1]
    start_value = 0
    print(path, 'xx', vi)
    for i in range(len(path)):
        result.append(nodeid(path[i][0], path[i][1]))
    return [result, vi]


def search(maze, cost, start, end):
    start_node = Node(None, tuple(start))
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, tuple(end))
    end_node.g = end_node.h = end_node.f = 0
    yet_to_visit_list = []
    visited_list = []
    visited_list1 = []

    # Add the start node
    yet_to_visit_list.append(start_node)
    outer_iterations = 0
    max_iterations = (len(maze) // 2) ** 10

    move = [[-1, 0],  # go up
            [0, -1],  # go left
            [1, 0],  # go down
            [0, 1]]

    no_rows, no_columns = ROW, COL

    # Loop until you find the end

    while len(yet_to_visit_list) > 0:
        outer_iterations += 1

        current_node = yet_to_visit_list[0]
        current_index = 0
        for index, item in enumerate(yet_to_visit_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index
        if outer_iterations > max_iterations:
            print("giving up on pathfinding too many iterations")
            return return_path(current_node, maze, visited_list1)

        yet_to_visit_list.pop(current_index)
        visited_list.append(current_node)
        visited_list1.append(current_node.position)
        if current_node == end_node:
            return return_path(current_node, maze, visited_list1)
        children = []

        for new_position in move:
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            if (node_position[0] > (no_rows - 1) or
                    node_position[0] < 0 or
                    node_position[1] > (no_columns - 1) or
                    node_position[1] < 0):
                continue
            if maze[node_position[0]][node_position[1]] != 0:
                continue
            new_node = Node(current_node, node_position)
            children.append(new_node)
        for child in children:
            if len([visited_child for visited_child in visited_list if visited_child == child]) > 0:
                continue
            child.g = current_node.g + cost
            child.h = (((child.position[0] - end_node.position[0]) ** 2) +
                       ((child.position[1] - end_node.position[1]) ** 2))

            child.f = child.g + child.h
            if len([i for i in yet_to_visit_list if child == i and child.g > i.g]) > 0:
                continue

            # Add the child to the yet_to_visit list
            yet_to_visit_list.append(child)
