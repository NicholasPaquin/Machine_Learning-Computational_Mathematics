import copy
import uuid
from collections import Iterator


class Graph:
    # maybe only have end node and find start nodes from that
    def __init__(self, end_node):
        # assert(type(start_nodes) == "list")
        self.start_nodes = []
        self.start_id = -1
        self.end_node = end_node
        print("initialized graph")
        self.initialize()

    # might get rid of forward step in favour of recursion experimenting a bit I suppose
    def forward(self, values, log=False):
        # current level of the graph being computed
        c_level = self.start_id
        # a que of all the nodes that need to be evaluated
        que = Que(self.start_nodes)

        # a dictionary of uuid's and the value to be passed onto a node of a given id
        values = self.value_dict(values)
        for node in que:
            print(node)
            if node.id == c_level:
                val, next_node = node.forward(values[node()])
                print(val)
                if not log:
                    del(values[node()])
                if next_node() not in values:
                    values[next_node()] = [val]
                else:
                    values[next_node()].append(val)
                # que.remove(node)
                if next_node not in que:
                    que.insert(next_node)
            # que.remove(node)
            # finish this once I'm done node class

    # returns a dictionary of uuids and values to be calculated for for each node
    def value_dict(self, values):
        assert(len(values) == len(self.start_nodes))
        dict = {}
        for i in range(len(values)):
            dict[self.start_nodes[i].uuid] = [values[i]]
        return dict

    # defines each level of the node and finds the starting nodes from the ending node
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
            next_nodes = self.clean(temp_next)

            if len(next_nodes) == 0:
                self.start_id = ind
                cataloged = True
            ind += 1
        print("Indexed nodes")

    def clean(self, list):
        return [elem for elem in list if elem]


class Que():
    def __init__(self, nodes):
        self.que = nodes

    def __getitem__(self, item):
        return self.que[item]

    def __iter__(self):
        if hasattr(self.que[0], "__iter__"):
            return self.que[0].__iter__()
        return self.que.__iter__()

    # insert in order
    def insert(self, node):
        for i in range(len(self.que)):
            if self.que[i].id < node.id:
                self.que.insert(node, i + 1)
                print("Inserted into que")
                return

    def remove(self, node):
        for i in range(len(self.que)):
            if self.que[i] == node:
                self.que.pop(i)
                return

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
        self.uuid = str(uuid.uuid1())

    def __eq__(self, other):
        return True if self.uuid == other.uuid else False

    def __call__(self, *args, **kwargs):
        return self.uuid

    def eq_uuid(self, uuid):
        return True if self.uuid == uuid else False

    def connect(self, nodes):
        self.last_nodes = nodes
        for node in nodes:
            node.next_node = self

    def forward(self, vars):
        assert (isinstance(vars, list))
        assert(len(vars) == self.variables)
        self.stored_val = self.operation(vars)
        return self.operation(vars), self.next_node

    def node_def(self):
        print(f"Variables: {self.variables}, Operation: {self.operation}, Next Node: {self.next_node}, UUID: {self.uuid}")

    def catalog(self, id):
        self.id = id


def assign(val):
    return val


def add(vals):
    sum = 0
    for i in vals:
        sum += i
    return sum


node1 = Node(1, assign)
node1.node_def()
node1.forward([7])
node2 = Node(1, assign)
node2.forward([7])
adder = Node(2, add)
adder.connect([node1, node2])
print(adder.forward([7, 7]))
graph = Graph(adder)
graph.forward([7, 7])

