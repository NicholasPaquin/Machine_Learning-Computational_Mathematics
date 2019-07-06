import copy


def remove_none(list):
    return [elem for elem in list if elem]


class Graph:
    # maybe only have end node and find start nodes from that
    def __init__(self, end_node):
        # assert(type(start_nodes) == "list")
        self.start_nodes = []
        self.end_node = end_node
        print("initialized graph")

    def forward(self, values):
        ind = 0
        for node in self.start_nodes:
            val, next_node = node.forward(values[ind])
            # do this recursively i think
            ind += 1
            # finish this once I'm done node class

    def initialize(self):
        self.end_node.catalog(-1)
        next_nodes = self.end_node.last_nodes
        ind = 0
        cataloged = False
        while not cataloged:
            temp_next = []
            for nodes in next_nodes:
                nodes.catalog(ind)
                temp_next.append(nodes.last_nodes)
                if not nodes.last_nodes:
                    self.start_nodes.append(nodes)
            next_nodes = remove_none(temp_next)
            ind += 1
            if len(next_nodes) == 0:
                cataloged = True
        print("Indexed nodes")



class Node:
    # same operation is preformed on all inputs
    # variables is the number of inputs to take, operation is the operation to preform on inputs
    # next node is a "pointer" to the next operational node
    def __init__(self, variables, operation, next_node=None, last_node=None):
        self.variables = variables
        self.operation = operation
        self.next_node = next_node
        self.last_nodes = last_node
        self.stored_val = 0.0
        self.id = -99

    def connect(self, nodes):
        self.last_nodes = nodes
        for node in nodes:
            node.next_node = self

    def forward(self, vars):
        # assert (type(vars) == "list")
        # assert(len(vars) == self.variables)
        self.stored_val = self.operation(vars)
        return self.operation(vars), self.next_node

    def node_def(self):
        print(f"Variables: {self.variables}, Operation: {self.operation}, Next Node: {self.next_node}")

    def catalog(self, id):
        self.id = id


def assign(val):
    return val


def add(vals):
    for i in vals:
        sum += i
    return sum


node1 = Node(1, assign)
node1.node_def()
node1.forward(7)
node2 = Node(1, assign)
node2.forward(7)
adder = Node(2, add)
adder.connect([node1, node2])
graph = Graph(adder)
graph.initialize()
print(graph.start_nodes)
