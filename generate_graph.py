"""
This document is used to synamically generate graphs that are specifically designed for my machine learning library
"""

class _Graph:
    def __init__(self, model):
        self.parameters = model.parameters
        self.start_node = None
        self.end_node = None

    def generate(self):
        for param in self.parameters:
            pass

