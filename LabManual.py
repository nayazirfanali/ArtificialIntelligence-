Artificial Intelligence for Developing Generative AI Applications
CSA – 1751
Shaik Nayaz Irfan Ali – 192211326

1.Write a Python program to solve 8 queens’ problem.
Aim:
To write a python program to solve 8 Queens Problem
Algorithm:
1.Start with an empty chessboard (8x8 grid).
2.Place a queen in the first column:
Start by placing a queen in row 1 of column 1.
3.Move to the next column:
For the next column, try placing a queen in any row that is not under attack (i.e., no other queen in the same row, column, or diagonal).
4.Check if the current placement is safe:
If a queen can be safely placed in the current column, place it and move to the next column.
If no safe row is found in the current column, backtrack (i.e., remove the last placed queen from the previous column and try a different row).
5.Repeat steps 3-4 until:
All 8 queens are placed on the board, or Backtrack all the way to the first column if no solution is found.
6.If all queens are placed:
Print or store the solution. 
7.If no valid placement is found after trying all rows and columns, declare that no solution exists.
file writer is a class
my writer is a object for that class
file writer class is available in java.io is a package
package means we should search for import only 
include try and catch with exception
program:
def print_board(board):
    for row in board:
        print(" ".join(row))
    print("\n")

def is_safe(board, row, col):
    n = len(board)

    for i in range(col):
        if board[row][i] == 'Q':
            return False

    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 'Q':
            return False

    for i, j in zip(range(row, n, 1), range(col, -1, -1)):
        if board[i][j] == 'Q':
            return False

    return True

def solve_n_queens(board, col):
    n = len(board)

    if col >= n:
        print_board(board)
        return True

    res = False
    for i in range(n):
        if is_safe(board, i, col):
            board[i][col] = 'Q'
            res = solve_n_queens(board, col + 1) or res
            board[i][col] = '.'

    return res

def solve_8_queens():
    n = 8
    board = [['.' for _ in range(n)] for _ in range(n)]

    if not solve_n_queens(board, 0):
        print("No solution exists")

solve_8_queens()

OUTPUT:

Q . . . . . . .       
. . . . . . Q .
. . . . Q . . .
. . . . . . . Q
. Q . . . . . .
. . . Q . . . .
. . . . . Q . .
. . Q . . . . .  

Result:
The above python program has been executed successfully.

2.Write a python program to solve water jug problem.
Aim:
To solve water jug problem using Python program.
Algorithm:
1.Start with two jugs of different capacities: Jug X with capacity x liters, and Jug Y with capacity y liters.
2.Goal: Obtain exactly z liters of water in either of the jugs.
3.Allowed operations:
Fill any of the jugs completely. Empty any of the jugs.
Pour water from one jug to the other until one jug is empty or the other jug is full.
4.Use BFS (Breadth-First Search) to explore all possible states of water in the jugs:
Maintain a queue of states (amount_in_X, amount_in_Y). Perform all possible operations on the jugs for each state. Keep track of visited states to avoid cycles.
5.Stop when exactly z litters is achieved in either jug or both combined.
Program:
from collections import deque

def is_solvable(x, y, z):
    if z > max(x, y) or z % gcd(x, y) != 0:
        return False
    return True

def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

def water_jug(x, y, z):
    if not is_solvable(x, y, z):
        return "No solution"

    visited = set()
    queue = deque([(0, 0)])

    while queue:
        a, b = queue.popleft()

        if a == z or b == z or a + b == z:
            return f"Solution found: Jug X = {a}, Jug Y = {b}"

        if (a, b) in visited:
            continue

        visited.add((a, b))

        queue.append((x, b))  # Fill jug X
        queue.append((a, y))  # Fill jug Y
        queue.append((0, b))  # Empty jug X
        queue.append((a, 0))  # Empty jug Y
        queue.append((max(0, a - (y - b)), min(y, a + b)))  # Pour from X to Y
        queue.append((min(x, a + b), max(0, b - (x - a))))  # Pour from Y to X

    return "No solution"

x = 4
y = 3
z = 2

print(water_jug(x, y, z))

OUTPUT:
Solution found: Jug X = 4, Jug Y = 2

Result:
The above python program has been executed successfully.


3.Write a python program to solve Crypt Arithmetic problem.
Aim:
To solve Crypt Arithmetic problem using Python programming.
Algorithm:
1.Cryptarithmetic Problem: 
Given an equation like SEND + MORE = MONEY, assign digits to letters such that the equation holds true.
2.Objective:
Each letter should represent a unique digit, and the resulting equation should be mathematically valid.
3.Approach:
Create all possible digit assignments for letters. Check each assignment to see if it satisfies the equation. Return the valid solution.
Program:
from itertools import permutations

def is_valid_solution(s1, s2, s3, mapping):
    num1 = int("".join(str(mapping[c]) for c in s1))
    num2 = int("".join(str(mapping[c]) for c in s2))
    num3 = int("".join(str(mapping[c]) for c in s3))
    return num1 + num2 == num3

def solve_cryptarithmetic(s1, s2, s3):
    letters = set(s1 + s2 + s3)
    if len(letters) > 10:
        return "No solution"

    for perm in permutations(range(10), len(letters)):
        mapping = dict(zip(letters, perm))
        if is_valid_solution(s1, s2, s3, mapping):
            return {letter: mapping[letter] for letter in letters}

    return "No solution"

s1 = "SEND"
s2 = "MORE"
s3 = "MONEY"

print(solve_cryptarithmetic(s1, s2, s3))

OUTPUT:
{'O': 0, 'R': 8, 'Y': 2, 'S': 9, 'D': 7, 'E': 5, 'M': 1, 'N': 6} 
Result:
The above python program has been executed successfully.

4.Write a Python program to implement BFS.
Aim:
To implement Breadth First Search using python programming. 
Algorithm:
1.Start at the source node.
2.Enqueue the source node to a queue.
3.Mark the source node as visited.
4.While the queue is not empty:
Dequeue a node from the front of the queue.
Process the dequeued node.
For each unvisited neighbor of the dequeued node:
Mark it as visited. Enqueue the neighbor.
5.Repeat until the queue is empty.
Program:
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)

    while queue:
        node = queue.popleft()
        print(node, end=" ")

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

graph = {
    'A': ['B', 'C', 'D'],
    'B': ['E', 'F'],
    'C': ['G'],
    'D': ['H'],
    'E': [],
    'F': [],
    'G': [],
    'H': []
}

bfs(graph, 'A')

OUTPUT:
A B C D E F G H
Result:
The above python program has been executed successfully.

5.Write a Python program to implement DFS.	
Aim:
To Implement Depth First Search using python programming.
Algorithm:
1.Start at the source node.
2.Mark the source node as visited.
3.Explore each neighbor of the current node:
For each unvisited neighbor, recursively apply DFS.
4.Repeat the process until all nodes are visited.
Program:
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    print(start, end=" ")
    
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

graph = {
    'A': ['B', 'C', 'D'],
    'B': ['E', 'F'],
    'C': ['G'],
    'D': ['H'],
    'E': [],
    'F': [],
    'G': [],
    'H': []
}

dfs(graph, 'A')

OUTPUT:
A B E F C G D H
Result:
The above python program has been executed successfully.

6.Write a Python program to implement A* Search.
Aim:
To Implement A* Search using python programming.
Algorithm:
1.Initialize the open list with the start node and set its cost g to 0.
2.Initialize the closed list as empty.
3.While the open list is not empty:
Find the node with the lowest f value (f = g + h) in the open list.
If the current node is the goal, return the path.
Move the current node from the open list to the closed list.
For each neighbor of the current node:
If the neighbor is already in the closed list, skip it.
If the neighbor is not in the open list or a shorter path is found:
Update its g and f values.
Set the current node as its parent.
Add it to the open list if not already there.
4.If the open list is empty and no path is found, return failure.
Program:
import heapq

def a_star_search(graph, start, goal, h):
    open_list = []
    heapq.heappush(open_list, (0 + h[start], start))

    g = {start: 0}
    parents = {start: None}
    closed_list = set()

    while open_list:
        current_f, current_node = heapq.heappop(open_list)

        if current_node == goal:
            path = []
            while current_node:
                path.append(current_node)
                current_node = parents[current_node]
            return path[::-1]

        closed_list.add(current_node)

        for neighbor, cost in graph[current_node]:
            if neighbor in closed_list:
                continue

            tentative_g = g[current_node] + cost
            if neighbor not in g or tentative_g < g[neighbor]:
                g[neighbor] = tentative_g
                f = tentative_g + h[neighbor]
                heapq.heappush(open_list, (f, neighbor))
                parents[neighbor] = current_node

    return None

graph = {
    'A': [('B', 1), ('C', 3)],
    'B': [('D', 3), ('E', 1)],
    'C': [('F', 5)],
    'D': [('G', 1)],
    'E': [('G', 4)],
    'F': [('G', 2)],
    'G': []
}

h = {
    'A': 6,
    'B': 4,
    'C': 4,
    'D': 2,
    'E': 2,
    'F': 2,
    'G': 0
}

path = a_star_search(graph, 'A', 'G', h)
print("Path:", path)

OUTPUT:
Path: ['A', 'B', 'D', 'G']\]]\\\]\]
Result:
The above python program has been executed successfully.


7.Write a python program to implement map colouring for attaining CSP.
Aim:
To Implement Map colouring for attaining CSP by using Python programming
Algorithm:
1.Start with an uncolored map.
2.Assign the first color to the first region.
3.Move to the next region:
Assign the smallest possible color that satisfies the constraint (no neighboring regions have the same color).
4.Backtrack if no valid color can be assigned to a region.
5.Repeat until all regions are colored.
Program:
def is_safe(graph, colors, region, color):
    for neighbor in graph[region]:
        if colors[neighbor] == color:
            return False
    return True

def map_coloring(graph, m, colors, region):
    if region == len(graph):
        return True

    for color in range(1, m + 1):
        if is_safe(graph, colors, region, color):
            colors[region] = color
            if map_coloring(graph, m, colors, region + 1):
                return True
            colors[region] = 0
    return False

def solve_map_coloring(graph, m):
    colors = [0] * len(graph)
    if map_coloring(graph, m, colors, 0):
        return colors
    else:
        return "No solution"

graph = {
    0: [1, 2],
    1: [0, 2, 3],
    2: [0, 1, 3],
    3: [1, 2]
}

m = 3
result = solve_map_coloring(graph, m)
print("Solution:", result)

OUTPUT:
Solution: [1, 2, 3, 1]
Result:
The above python program has been executed successfully.

8.Write a python program to implementing a TIC TAC TOE game.
Aim:
To Implementing a TIC TAC TOE game by using Python programming.
Algorithm:
1.Initialize an empty 3x3 board.
2.Players take turns marking their symbol ('X' or 'O') on the board.
3.Check for a win after each move:
A player wins if they occupy an entire row, column, or diagonal.
4.Check for a tie if all positions are filled with no winner.
5.Repeat until either a player wins or the game ends in a tie.
Program:
def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 5)

def check_winner(board, player):
    for row in board:
        if all([s == player for s in row]):
            return True
    for col in range(3):
        if all([board[row][col] == player for row in range(3)]):
            return True
    if all([board[i][i] == player for i in range(3)]) or all([board[i][2-i] == player for i in range(3)]):
        return True
    return False

def check_draw(board):
    return all([cell != ' ' for row in board for cell in row])

def tic_tac_toe():
    board = [[' ' for _ in range(3)] for _ in range(3)]
    current_player = 'X'

    while True:
        print_board(board)
        row = int(input(f"Player {current_player}, enter row (0, 1, 2): "))
        col = int(input(f"Player {current_player}, enter col (0, 1, 2): "))

        if board[row][col] == ' ':
            board[row][col] = current_player
        else:
            print("Cell already taken, try again.")
            continue

        if check_winner(board, current_player):
            print_board(board)
            print(f"Player {current_player} wins!")
            break

        if check_draw(board):
            print_board(board)
            print("It's a draw!")
            break

        current_player = 'O' if current_player == 'X' else 'X'

tic_tac_toe()

OUTPUT:
Player O, enter col (0, 1, 2): 1
  |   | X
-----
  |   |  
-----
  | O |  
-----
Player X, enter row (0, 1, 2): 0
Result:
The above python program has been executed successfully.

9.Write a python program to implement Travelling sales men program.
Aim:
To Implement Travelling sales men program by using Python programming.
Algorithm:
1.Define a graph with n nodes and a distance matrix.
2.Use dynamic programming to explore all possible subsets of nodes.
3.For each subset, calculate the minimum cost to visit all nodes in the subset and return to the starting node.
4.Store intermediate results to avoid recalculating costs for the same subset.
5.Return the minimum cost for visiting all nodes and completing the tour.
Program:
import itertools

def travelling_salesman_problem(graph, start):
    n = len(graph)
    all_nodes = range(n)

    dp = {}

    # Initialize dp table for single-node subsets excluding the start node
    for subset_size in range(1, n):
        for subset in itertools.combinations(all_nodes, subset_size):
            if start in subset:
                continue
            dp[(subset, subset[0])] = float('inf')

    # Set the starting point, no cost to start at the starting node
    dp[((start,), start)] = 0

    # Fill dp table for larger subsets
    for subset_size in range(2, n + 1):
        for subset in itertools.combinations(all_nodes, subset_size):
            if start not in subset:
                continue
            for k in subset:
                if k == start:
                    continue
                prev_subset = tuple([x for x in subset if x != k])
                dp[(subset, k)] = float('inf')
                for m in prev_subset:
                    if (prev_subset, m) in dp:
                        dp[(subset, k)] = min(dp[(subset, k)], dp[(prev_subset, m)] + graph[m][k])

    # Calculate the minimum cost to complete the cycle
    final_res = min(dp[(tuple(all_nodes), k)] + graph[k][start] for k in range(1, n))

    return final_res

graph = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

start_node = 0
result = travelling_salesman_problem(graph, start_node)
print("Minimum cost:", result)

OUTPUT:
Minimum cost: 80
Result:
The above python program has been executed successfully.

