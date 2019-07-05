import copy

class Graph:
    def __init__(self, start_nodes, end_node):
        assert(type(start_nodes) == "list")
        self.start_nodes = start_nodes
        self.end_node = end_node
        print("initialized graph")

    def forward(self, values):
        ind = 0
        for node in self.start_nodes:
            val, next_node = node.forward(values[ind])
            ind += 1
            # finish this once I'm done node class

    def id(self):
        self.end_node.catalog(-1)
        next_nodes = self.end_node.last_nodes
        ind = 0
        qued = False
        while not qued:
            for nodes in next_nodes:
                nodes.catalog(ind)
            ind += 1




class Node:
    # same operation is preformed on all inputs
    # variables is the number of inputs to take, operation is the operation to preform on inputs
    # next node is a "pointer" to the next operational node
    def __init__(self, variables, operation, next_node, last_node):
        self.variables = variables
        self.operation = operation
        self.next_node = next_node
        self.last_nodes = last_node
        self.id = 0

    def forward(self, vars):
        assert (type(vars) == "list")
        assert(len(vars) == self.variables)
        return self.operation(vars), self.next_node

    def node(self):
        print(f"Variables: {self.variables}, Operation: {self.operation}, Next Node: {self.next_node}")

    def catalog(self, id):
        self.id = id



