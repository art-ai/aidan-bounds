import itertools
import heapq
from timer import Timer

PLOT_SEARCH_SPACE = False

class ThresholdTest:
    id_counter = 0 # next available id
    def __init__(self, weights, threshold, size=None, bounds=None):
        #establishing variables
        self.weights = weights
        self.threshold = threshold

        # instead of copying weights over and over again,
        # we use the same weights list and just vary the size
        if size is None:
            self.size = len(weights)
        else:
            self.size = size

        # it is faster to update the bounds externally instead of
        # re-computing them each time
        if bounds is None:
            self.bounds = Bounds.from_weights(weights,size)
        else:
            self.bounds = bounds

        self.id = ThresholdTest.new_id()

    @classmethod
    def new_id(cls):
        new_id = cls.id_counter
        cls.id_counter += 1
        return new_id

    def __lt__(self,other):
        return self.id < other.id
    
    def __repr__(self):
        #printing threshold
        # Creates a representation of the threshold test inside of the tree
        weights = self.get_weights()
        if not weights:
            return f"0 &#8805; {self.threshold}"
        root = " + ".join([f"{weight}*x<SUB>{var+1}</SUB>" for var, weight in enumerate(weights)])
        root = f"{root} &#8805; {self.threshold}"
        return root

    def get_weights(self):
        # try not to use this function
        return self.weights[:self.size]

    def get_last_weight(self):
        # note that weights may be longer than size
        # (this is to avoid re-copying the weights in set_last_input()
        return self.weights[self.size-1]

    # Functions testing for triviality:
    def is_trivial_fail(self):
        lower,upper = self.bounds.get_bounds()
        threshold = self.threshold
        # True if the lower and upper are both less than the threshold
        return lower <= upper < threshold

    def is_trivial_pass(self):
        lower,upper = self.bounds.get_bounds()
        threshold = self.threshold
        # True if both the lower and upper are greater than the threshold
        return threshold <= lower <= upper
  
    # Form_tree (pruned tree) is based off this truth table
    def as_truth_table(self):
        passed = 0
        fail = 0
        pass_list = [0]
        fail_list = [0]

        all_combinations = list(itertools.product([0, 1], repeat=self.size))
        headers = [f"X_{i + 1}" for i in (range(self.size))]
        #print("|".join(headers) + "|Result")

        weights = self.get_weights()

        #iterating through all possible combinations
        for c in all_combinations:
            #sums the weighted inputs from the combinations
            weighted_sum = sum(weight * inputs for weight, inputs in zip(weights, c))
            #satisfied variable returns True if test is passed and False otherwise
            satisfied = weighted_sum >= self.threshold

            # With this passes increase while fails remain stagnant, and vice versa
            if satisfied:
                passed += 1
                pass_list.append(passed)
                fail_list.append(fail)
            else:
                fail += 1
                fail_list.append(fail)
                pass_list.append(passed)
        
        # Returns list of passes and fails from the truth table
        return pass_list, fail_list
    
    # Function used to compute the threshold test
    def set_last_input(self, value):
        assert self.size > 0
        last_weight = self.get_last_weight()

        # update threshold
        #when values in path are set to 1
        if value == 1:
            new_threshold = self.threshold - last_weight
        #when values in path are set to 0
        elif value == 0:
            new_threshold = self.threshold

        # update bounds
        lb,ub = self.bounds.get_bounds()
        if last_weight > 0:
            ub -= last_weight
        else:
            lb -= last_weight
        bounds = Bounds(lb,ub)

        return ThresholdTest(self.weights, new_threshold, size=self.size-1, bounds=bounds)

# Class used to measure the upper and lower bounds of the threshold test    
class Bounds():
    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub

    @classmethod
    def from_weights(cls,weights,size):
        lb,ub = 0,0
        for weight in weights[:size]:
            if weight < 0: lb += weight
            if weight > 0: ub += weight
        return cls(lb,ub)

    def upper_bound(self):
        return self.ub

    def lower_bound(self):
        return self.lb

    def get_bounds(self):
        return self.lb,self.ub

    # Measures the differences of the upper and lower bound from the respective threshold
    def gap_size(self, test):

        upper_diff = abs(self.upper_bound() - test.threshold)
        lower_diff = abs(test.threshold - self.lower_bound())

        score_list = [self.lower_bound(), lower_diff, test.threshold,upper_diff, self.upper_bound()]
        return score_list

# Functions previously used to display the computations the threshold test was making
def print_path(test, values, depth=0):

    print("  " * depth + str(test))
    if not values:
        return

    next_value = values[-1]
    print("  " * depth + f"└-[x_{test.size}={next_value}]- ", end="")

    reduced_test = test.set_last_input(next_value)
    print_path(reduced_test, values[:-1], depth + 1)
    # ideally recursive

def print_tree(test, depth=0):
    print("  " * depth + str(test))
    if test.size == 0:
        return

    next_value = 0
    reduced_test = test.set_last_input(next_value)
    print_tree(reduced_test, depth + 1)
    next_value = 1
    reduced_test = test.set_last_input(next_value)
    print_tree(reduced_test, depth + 1)

class NullPlotter:
    def __init__(self): pass
    def add_node(self, node_id, label, color): pass
    def add_edge(self,  parent_id, child_id, label): pass
    def draw_tree(self, filename="tree_plot.png"): pass

# Class used for the making of the decision tree, utilizes pygraphviz
class TreePlotter():
    def __init__(self):
        import pygraphviz as pgv
        # Initializing a graph, in this case being a strict one
        self.graph = pgv.AGraph(strict=True, directed=False)

    # Function used to add nodes to the tree:
    def add_node(self, node_id, label, color):
        self.graph.add_node(node_id, label=f"<{label}>", shape='box', color=color)

    # Function to add edges to the nodes, from parent to child
    def add_edge(self,  parent_id, child_id, label):
        value = label[-1]
        style = "solid" if value == "1" else "dashed"

        self.graph.add_edge(parent_id, child_id, label=f"<{label}>", style = style)
        
       # edge_count = 0
       # edge_count += 1
    # Draws the actual tree and saves the image as a .png file
    def draw_tree(self, filename="tree_plot.png"):
        self.graph.layout(prog="dot")
        self.graph.draw(filename)

# Class used to create an array of pass and fails 
class Counter():
    def __init__(self,test_size):
        # initializing variables
        self.test_size = test_size
        self.passes = 0
        self.fails = 0
        self.pass_counts = [0]
        self.fail_counts = [0]
    
    # Using previous functions of triviality, counts passes and fails
    def is_trivial_and_count(self,test):
        total_counts = 2 ** test.size
        if test.is_trivial_pass():
            self.passes += total_counts
            self.pass_counts.append(self.passes)
            self.fail_counts.append(self.fails)
            return True
        if test.is_trivial_fail():
            self.fails += total_counts
            self.fail_counts.append(self.fails)
            self.pass_counts.append(self.passes)
            return True
        return False

# Function used to create the pruned tree using a depth-first search algorithm
def form_tree(plot, test, parent_id=None, depth=0, counter=None):
    
    # Displaying important values onto the nodes (mainly steps to pass and fail)
    #pass_steps, fail_steps = steps_to_pass(test), steps_to_fail(test)

    #bounds = Bounds(test.weights)
    #count = 2 ** test.size
    #current_label = f"{test}\n{bounds.gap_size(test)}\n{pass_steps}\n{fail_steps}\nCount: {count}"
    current_id = f"Node_{depth}_{test.threshold}"
    if test.is_trivial_pass():
        node_color = 'green'
    elif test.is_trivial_fail():
        node_color = 'red'
    else:
        node_color = 'black'
    #plot.add_node(current_id, current_label, color = node_color)

    #if parent_id is not None:
        #plot.add_edge(parent_id, current_id, label=f"x_{test.size + 1}")

    if counter.is_trivial_and_count(test):
        return

    # Iterates between the binary inputs of 0 and 1
    for next_value in [1, 0]:
        # Recursion, adds upon the depth of the tree
        reduced_test = test.set_last_input(next_value)
        form_tree(plot, reduced_test,parent_id=current_id, depth=depth+1, counter=counter)
    
    #plot.draw_tree("tree_plot.png")

# Improved form_tree function using a best-first search algorithm
# with the use of heaps
# NOTE: Will be going through this function step-by-step



def bfs_form_tree(plot, test, threshold, parent_id=None, depth=0, counter=None):
    # Initializing an empty heap array we will be iterating upon:
    heap = []
    # after pop, key is created which is the depth, then the threshold, (depth, threshold) , the value of the key is the threshold test
    # check key, if not there then add to cache, if there then do not check or add, just add edge (if plot search space)
    seen = {}

    # Creating and "pushing" our priorities to the heap
    iteration = 0

    def compute_priority(test):
        return min(steps_to_pass(test), steps_to_fail(test))

    initial_priority = compute_priority(test)
    heapq.heappush(heap, (initial_priority, depth, test, parent_id, None))

    # "while" The heap has values within itself
    while heap:
        # Removes the smallest of these variables from the heap and returns it
        priority, depth, test, parent_id, edge_label = heapq.heappop(heap)
                
        key = (depth, test.threshold)
        
        # Check if test has already been processed at this depth
        if key in seen:
            current_id = f"Node_{depth}_{test.id}"

            if PLOT_SEARCH_SPACE:
                current_label = f"{test} (\nIter. {iteration})"
                plot.add_node(current_id, current_label, color='gray')
                if parent_id is not None:
                    plot.add_edge(parent_id, current_id, label=f"x<SUB>{test.size + 1}</SUB> = {edge_label}")
            iteration += 1
            continue  # Skip processing — already seen

        seen[key] = test  # Mark this (depth, threshold) combo as seen
        
        current_id = f"Node_{depth}_{test.threshold}"        
    
        # Setting colors of the nodes based off passing or failing
        if test.is_trivial_pass(): 
            node_color = 'green'
        elif test.is_trivial_fail():       
            node_color = 'red'
        # The bread and butter of this function. Using how close a test is to triviality, takes priority upon
        # the node that is closer (in this case being closer to passing) and pushes that value onto the heap
        else:
            node_color = 'black'
            
            left = test.set_last_input(0)
            right = test.set_last_input(1)
            left_priority = compute_priority(left)
            right_priority = compute_priority(right)

            heapq.heappush(heap, (left_priority,  depth+1, left,  current_id, f"0"))
            heapq.heappush(heap, (right_priority, depth+1, right, current_id, f"1"))

        if PLOT_SEARCH_SPACE:
            current_label = f"{test} (\nIter. {iteration})"
            # Adding nodes to tree
            plot.add_node(current_id, current_label, color=node_color)

            # While there are still parents within the tree, add edges
            if parent_id is not None:
                plot.add_edge(parent_id, current_id, label=f"x<SUB>{test.size + 1}</SUB> = {edge_label}")

        iteration += 1
        # When the test/node is trivial, continue and don't make computations upon the already trivial test
        if counter.is_trivial_and_count(test):
            continue
        # ^^Function continues until heap is empty^^        
    
    # Draws the tree
    plot.draw_tree("tree_plot.png")

# Iterates throught the amount of steps a node is away from becoming trivial
# using the set_last_input to check
def steps_to_pass(test):
    weights = test.weights
    T = test.threshold
    size = test.size

    lb,ub = test.bounds.get_bounds()
    steps = 0
    while True:
        if T <= lb: return steps
        steps += 1
        weight = weights[size-steps]
        if weight < 0:
            # set input to 0
            lb -= weight
        else:
            # set input to 1
            T -= weight
            ub -= weight

def steps_to_fail(test):
    weights = test.weights
    T = test.threshold
    size = test.size

    lb,ub = test.bounds.get_bounds()
    steps = 0
    while True:
        if ub < T: return steps
        steps += 1
        weight = weights[size-steps]
        if weight < 0:
            # set input to 1
            T -= weight
            lb -= weight
        else:
            # set input to 0
            ub -= weight

# Graph showing the upper and lower bounds of the threshold test, and the steps taken to reach that number
# Mainly going to be used to show the efficiency of our algorithm

# The BFS, DFS and raw values are taken as parameters
def pass_fail_graph(bfs_pass, bfs_fail, pass_list, fail_list, pruned_pass, pruned_fail, test):
    import matplotlib
    from matplotlib import pyplot as plt

    font_size = 20

    matplotlib.rcParams.update({'xtick.labelsize': font_size,
                                'ytick.labelsize': font_size,
                                'figure.autolayout': True})
    font = {'family': 'sans-serif',
            'weight': 'normal',
            'size': 15}
    matplotlib.rc('font', **font)
    # computing the top value (this is 2^n)
    total = 2 ** test.size
    
    # Flipping the fail_list to start at the top and count down
    pruned_fail = [total - fail for fail in pruned_fail]
    fail_list = [total - fail for fail in fail_list]
    bfs_fail = [total - fail for fail  in bfs_fail]
    
    # Makes sure pass_list and flipped fail_list meet at the same endpoint
    assert pass_list[-1] == fail_list[-1]
    
    # Final count of the lists
    count = pass_list[-1]
   
    plt.plot(pass_list,color='blue',linestyle='-')
    plt.plot(fail_list,color='red', linestyle='-')
    plt.plot(pruned_fail,color='red' , linestyle= '-')
    plt.plot(pruned_pass,color='blue' , linestyle= '-')
    plt.plot(bfs_pass, color='magenta', linestyle='-')
    plt.plot(bfs_fail,color='purple', linestyle='-') 
    plt.axhline(y=count, linestyle='--', color="black")

    plt.show()


#weights = [-1,-2, 2, 4, 8, -11]
#weights = [-2, 3, -4, 5]
#weights = [-30, 4, 8, 22, 9, 12, -17]
#weights = [-12, 15, -8, 6, -23, 30, -4, 18, -9, 11]
#weights = [1, -1, 2, -2, 4,-4, 8 -8, 3, -2, 1]
#weights = [64,-64,32,-32,16,-16,8,-8,4,-4,2,-2,1,-1]
#weights = [1024,-1024,512,-512,256,-256,128,-128,64,-64,32,-32,16,-16,8,-8,4,-4,2,-2,1,-1]
n = 20
weights = [ 2**x for x in range(n) ] + [ -2**x for x in range(n) ]
weights = sorted(weights,key=lambda x: abs(x))
threshold = 1
threshold_test = ThresholdTest(weights, threshold)

if PLOT_SEARCH_SPACE: plotter = TreePlotter()
else:                 plotter = NullPlotter()

with Timer("dfs"):
    counter = Counter(threshold_test.size)
    form_tree(plotter, threshold_test, counter=counter)
    pFail_list, pPass_list = counter.fail_counts, counter.pass_counts

#with Timer("truth table"):
#    pass_list, fail_list = threshold_test.as_truth_table()
pass_list, fail_list = [pPass_list[-1]],[pFail_list[-1]]

with Timer("bfs"):
    bfs_counter = Counter(threshold_test.size)
    bfs_form_tree(plotter, threshold_test, threshold, counter=bfs_counter)
    bfsFail_list, bfsPass_list = bfs_counter.fail_counts, bfs_counter.pass_counts

pass_fail_graph(bfsPass_list,bfsFail_list,pass_list, fail_list, pPass_list, pFail_list, threshold_test)
