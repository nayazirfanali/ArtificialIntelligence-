**1) 8 queens’ problem. - Algorithm**
 1.Start with an empty board: Imagine an 8x8 chessboard with all squares empty. 2.Place a Queen: Choose a row for the first queen. 3.Check for Conflicts: Check if placing the queen in that row causes any conflicts with existing queens. If there's a conflict, move the queen to the next available row in the same column and repeat. If no conflict, proceed to the next queen. 4.Repeat: Continue placing queens one by one, ensuring no conflicts arise. 5.Backtrack: If you reach a point where you can't place a queen without conflicts, backtrack to the previous queen and try a different position. 6.Solution Found: If you successfully place all eight queens without conflicts, you've found a solution.

**2) water jug problem - Algorithm**
 1.Start with Empty Jugs: Begin with both jugs empty. 2.Fill or Empty: Choose one of the following actions: Fill one of the jugs to its capacity. Empty one of the jugs completely. Pour water from one jug to the other until either the source jug is empty or the destination jug is full. 3.Check Target: After each action, check if the target amount of water is in either jug. If so, you've solved the problem. 4.Repeat: If the target amount is not reached, repeat steps 2 and 3 until you find a solution or determine that no solution exists.

**3) Crypt Arithmetic problem = Algorithm**
 1.Constraint Analysis: Identify the constraints imposed by the equation. For example, if a letter appears in multiple places, it must represent the same digit. 2.Deduction: Use logical reasoning to deduce possible values for certain letters based on the constraints. For instance, if a letter appears in the units place of a sum, it must be a non-zero digit. 3.Trial and Error: Test different digit assignments for letters, ensuring they satisfy the constraints and the equation. 4.Backtracking: If a path leads to an inconsistent equation, backtrack and try a different assignment.

**4) To implement BFS - Algorithm**
 1.Initialize a queue: BFS uses a queue to explore nodes in the breadth-first order. A node is dequeued, its neighbors are visited, and they are then enqueued. 2.Mark all nodes as unvisited: Maintain a boolean array (visited[]) to keep track of whether a node has been visited. 3.Enqueue the starting node and mark it visited: The starting node is enqueued and marked as visited. 4.Dequeue a node: Dequeue a node from the queue, and print or process the node. 5.Visit all unvisited neighbors: For each unvisited neighbor of the dequeued node, mark it as visited and enqueue it. 6.Repeat until the queue is empty: The process continues until all nodes have been visited or the queue becomes empty.

**5) To Implement DFS - Algorithm**
 1.Start from a given node: Begin the traversal at the starting node. 2.Mark the current node as visited: Maintain a visited[] array to keep track of visited nodes. 3.Explore adjacent unvisited nodes: Recursively or iteratively visit each unvisited neighbor of the current node. 4.Backtrack: Once a node has no more unvisited neighbors, backtrack to explore other paths. 5.Repeat until all nodes are visited.

**6) A Star Search - Algorithm**
 1.Initialize the open list with the start node and set its cost g to 0. 2.Initialize the closed list as empty. 3.While the open list is not empty: Find the node with the lowest f value (f = g + h) in the open list. If the current node is the goal, return the path. Move the current node from the open list to the closed list. For each neighbor of the current node: If the neighbor is already in the closed list, skip it. If the neighbor is not in the open list or a shorter path is found: Update its g and f values. Set the current node as its parent. Add it to the open list if not already there. 4.If the open list is empty and no path is found, return failure.

**7) Map colouring for attaining CSP - Algorithm**
 1.Start with an uncolored map. 2.Assign the first color to the first region. 3.Move to the next region: Assign the smallest possible color that satisfies the constraint (no neighboring regions have the same color). 4.Backtrack if no valid color can be assigned to a region. 5.Repeat until all regions are colored.

**8) TIC TAC TOE game - Algorithm**
 1.Initialize an empty 3x3 board. 2.Players take turns marking their symbol ('X' or 'O') on the board. 3.Check for a win after each move: A player wins if they occupy an entire row, column, or diagonal. 4.Check for a tie if all positions are filled with no winner. 5.Repeat until either a player wins or the game ends in a tie.

**9) Travelling sales men program - Algorithm**
 1.Define a graph with n nodes and a distance matrix. 2.Use dynamic programming to explore all possible subsets of nodes. 3.For each subset, calculate the minimum cost to visit all nodes in the subset and return to the starting node. 4.Store intermediate results to avoid recalculating costs for the same subset. 5.Return the minimum cost for visiting all nodes and completing the tour.

**10) Alpha Beta Purinng - Algorithm**
1.Start with initial alpha (worst case for maximizer, initially -∞) and beta (worst case for minimizer, initially ∞).
2.Traverse the tree:
If the current node is a maximizing player, update alpha (maximum value encountered).
If the current node is a minimizing player, update beta (minimum value encountered).
3.If alpha >= beta, prune the remaining branches (skip evaluating them).
4.Continue recursively until the leaves are reached.

**11) Decission Tree - Algorithm**
1.Select the Best Attribute: The algorithm picks the best feature to split the data based on a splitting criterion (e.g., Gini Impurity or Information Gain).
2.Split the Dataset: Recursively split the dataset based on the best attribute.
3.Leaf Nodes: Once all data is perfectly classified, or a stopping criterion is met, the tree stops growing and assigns a class to the leaf nodes

**12) Feed Forward neural network - Alorithm**
1. Initialization:
Weights and Biases: Randomly initialize the weights and biases for each neuron in the network.
2. Forward Propagation:
Input Layer: Feed the input data to the input layer.
Hidden Layer:
Calculate the weighted sum of inputs and biases for each neuron.
Apply an activation function (e.g., sigmoid) to the result.
Output Layer:
Repeat the above steps for the hidden layer to get the final output.
3. Backpropagation:
Calculate Error: Compute the difference between the predicted output and the actual target.
Propagate Error: Propagate the error backward through the network, updating the weights and biases based on the error and the activation function's derivative.
4. Update Weights and Biases:
Adjust the weights and biases using gradient descent, taking a step in the direction that reduces the error.
5. Repeat:
Iterate through the training data multiple times (epochs), repeating steps 2-4 to refine the network's weights and biases.
6. Prediction:
Once the network is trained, feed new input data into the network using the same forward propagation steps to get the predicted output.

**13) 8-Puzzle Problem - Algorithm**
Define the start and goal states of the 8-puzzle.
Use BFS (Breadth-First Search) or A* algorithm to explore the state space.
For BFS: Enqueue the initial state and explore neighboring states.
For A*: Use a priority queue with a heuristic (e.g., Manhattan distance).
Continue until the goal state is reached.
Backtrack to find the solution path.

**14) Vacuum Cleaner Problem - Algorithm**
Represent the environment as a grid of rooms, each either clean or dirty.
Use a reflex agent or search-based method to explore the grid.
Clean a room if it's dirty, then move to the next room.
The goal is achieved when all rooms are clean.

**15) Missionaries and Cannibals Problem - Algorithm**
Represent the state as the number of missionaries and cannibals on each riverbank.
Use BFS or DFS to explore transitions, moving missionaries and cannibals.
Ensure that cannibals never outnumber missionaries on either side.
The goal is to get all the missionaries and cannibals across the river safely.





 
