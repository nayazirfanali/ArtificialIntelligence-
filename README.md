**1) 8 queens’ problem. - Algorithm**
1.Start with an empty board: Imagine an 8x8 chessboard with all squares empty.
2.Place a Queen: Choose a row for the first queen.
3.Check for Conflicts: Check if placing the queen in that row causes any conflicts with existing queens.
If there's a conflict, move the queen to the next available row in the same column and repeat.
If no conflict, proceed to the next queen.
4.Repeat: Continue placing queens one by one, ensuring no conflicts arise.
5.Backtrack: If you reach a point where you can't place a queen without conflicts, backtrack to the previous queen and try a different position.
6.Solution Found: If you successfully place all eight queens without conflicts, you've found a solution.

**2) water jug problem - Algorithm**
1.Start with Empty Jugs: Begin with both jugs empty.
2.Fill or Empty: Choose one of the following actions:
Fill one of the jugs to its capacity.
Empty one of the jugs completely.
Pour water from one jug to the other until either the source jug is empty or the destination jug is full.
3.Check Target: After each action, check if the target amount of water is in either jug. If so, you've solved the problem.
4.Repeat: If the target amount is not reached, repeat steps 2 and 3 until you find a solution or determine that no solution exists.

**3) Crypt Arithmetic problem = Algorithm**
1.Constraint Analysis: Identify the constraints imposed by the equation. For example, if a letter appears in multiple places, it must represent the same digit.
2.Deduction: Use logical reasoning to deduce possible values for certain letters based on the constraints. For instance, if a letter appears in the units place of a sum, it must be a non-zero digit.
3.Trial and Error: Test different digit assignments for letters, ensuring they satisfy the constraints and the equation.
4.Backtracking: If a path leads to an inconsistent equation, backtrack and try a different assignment.

**4) To implement BFS - Algorithm**
1.Initialize a queue: BFS uses a queue to explore nodes in the breadth-first order. A node is dequeued, its neighbors are visited, and they are then enqueued.
2.Mark all nodes as unvisited: Maintain a boolean array (visited[]) to keep track of whether a node has been visited.
3.Enqueue the starting node and mark it visited: The starting node is enqueued and marked as visited.
4.Dequeue a node: Dequeue a node from the queue, and print or process the node.
5.Visit all unvisited neighbors: For each unvisited neighbor of the dequeued node, mark it as visited and enqueue it.
6.Repeat until the queue is empty: The process continues until all nodes have been visited or the queue becomes empty.

**5) To Implement DFS - Algorithm**
1.Start from a given node: Begin the traversal at the starting node.
2.Mark the current node as visited: Maintain a visited[] array to keep track of visited nodes.
3.Explore adjacent unvisited nodes: Recursively or iteratively visit each unvisited neighbor of the current node.
4.Backtrack: Once a node has no more unvisited neighbors, backtrack to explore other paths.
5.Repeat until all nodes are visited.

**6) A Star Search - Algorithm**
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

**7) Map colouring for attaining CSP - Algorithm**
1.Start with an uncolored map.
2.Assign the first color to the first region.
3.Move to the next region:
Assign the smallest possible color that satisfies the constraint (no neighboring regions have the same color).
4.Backtrack if no valid color can be assigned to a region.
5.Repeat until all regions are colored.

**8) TIC TAC TOE game - Algorithm**
1.Initialize an empty 3x3 board.
2.Players take turns marking their symbol ('X' or 'O') on the board.
3.Check for a win after each move:
A player wins if they occupy an entire row, column, or diagonal.
4.Check for a tie if all positions are filled with no winner.
5.Repeat until either a player wins or the game ends in a tie.

**9) Travelling sales men program - Algorithm**
1.Define a graph with n nodes and a distance matrix.
2.Use dynamic programming to explore all possible subsets of nodes.
3.For each subset, calculate the minimum cost to visit all nodes in the subset and return to the starting node.
4.Store intermediate results to avoid recalculating costs for the same subset.
5.Return the minimum cost for visiting all nodes and completing the tour.
