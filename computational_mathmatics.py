import copy


def remove_none(list):
    return [elem for elem in list if elem]


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
        cataloged = False
        while not cataloged:
            temp_next = []
            for nodes in next_nodes:
                nodes.catalog(ind)
                temp_next.append(nodes.last_nodes)
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
        self.stored_var = 0.0
        self.id = 0

    def forward(self, vars):
        assert (type(vars) == "list")
        assert(len(vars) == self.variables)
        return self.operation(vars), self.next_node

    def node_def(self):
        print(f"Variables: {self.variables}, Operation: {self.operation}, Next Node: {self.next_node}")

    def catalog(self, id):
        self.id = id


def assign(val):
    return val


node = Node(1, assign)
node.node_def()