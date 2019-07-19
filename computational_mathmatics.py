import copy
import uuid
from collections import Iterator
from operations import *
import numpy as np


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
        print("Starting forward pass")
        # current level of the graph being computed
        c_level = self.start_id
        # a que of all the nodes that need to be evaluated
        que = Que(self.start_nodes)

        # a dictionary of uuid's and the value to be passed onto a node of a given id
        values = self.value_dict(values)
        for node in que:
            if node.id != c_level:
                c_level -= 1
            if node.id == c_level:
                val, next_nodes = node.forward(values[node()])

                if not log:
                    del(values[node()])
                for next_node in next_nodes:
                    if next_node:
                        if next_node() not in values:
                            print("Adding new element")
                            values[next_node()] = np.array(val)
                        else:
                            print("adding to existing element")
                            values[next_node()] = np.append(values[next_node()], np.array(val))
                        if next_node not in que:
                            que.insert(next_node)
                    else:
                        print(val)
                        return
            # que.remove(node)

    # returns a dictionary of uuids and values to be calculated for for each node
    def value_dict(self, values):
        assert(len(values) == len(self.start_nodes))
        dict = {}
        for i in range(len(values)):
            dict[self.start_nodes[i].uuid] = np.array(values[i])
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
                print(nodes)
                nodes.catalog(ind)
                temp_next.append(nodes.last_nodes)
                # one there are no nodes after a node, iut is considered a starting node
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


class Que:
    def __init__(self, nodes):
        self.que = nodes

    def __getitem__(self, item):
        return self.que[item]

    def __iter__(self):
        if hasattr(self.que[0], "__iter__"):
            return self.que[0].__iter__()
        return self.que.__iter__()

    def __len__(self):
        return len(self.que)

    # insert in order
    def insert(self, node):
        for i in range(len(self.que) - 1, -1, -1):
            if self.que[i].id > node.id:
                self.que.insert(i+1, node)
                return

    def remove(self, node):
        for i in range(len(self.que)):
            if self.que[i] == node:
                self.que.pop(i)
                print(f"Popped node {node()}")
                return


class Node:
    # same operation is preformed on all inputs
    # variables is the number of inputs to take, operation is the operation to preform on inputs
    # next node is a "pointer" to the next operational node
    def __init__(self, variables=None, operation=None, next_node=[None], last_node=None):
        # number of variables that operations will be preformed on
        self.variables = variables
        # operation type for node
        self.operation = operation
        # next node/s object
        self.next_nodes = next_node
        self.last_nodes = last_node
        # stores previously calculated value
        self.stored_val = 0.0
        # stores location in graph
        self.id = -99
        # for indentifying nodes apart from eachother
        self.uuid = str(uuid.uuid4())

    def __eq__(self, other):
        return True if self.uuid == other.uuid else False

    def __call__(self, *args, **kwargs):
        return self.uuid

    def eq_uuid(self, uuid):
        return True if self.uuid == uuid else False

    # connects a node to another or a set, sets current as parent to another node
    def connect(self, nodes):
        self.last_nodes = nodes
        for node in nodes:
            node.next_nodes = self.clean(node.next_nodes)
            node.next_nodes.append(self)

    def forward(self, vars: np.array):
        assert(vars.size == self.variables)
        self.stored_val = self.operation(vars)
        return self.operation(vars), self.next_nodes

    def node_def(self):
        return f"Variables: {self.variables}, Operation: {self.operation}, Next Node: {self.next_nodes[0]}, UUID: {self.uuid}"

    def catalog(self, id):
        self.id = id

    def clean(self, list):
        return [elem for elem in list if elem]

