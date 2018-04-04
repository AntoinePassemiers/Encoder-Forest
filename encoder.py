# -*- coding: utf-8 -*-
# encoder.py
# author : Antoine Passemiers

from rules import NodeRule, PathRule

import copy
import numpy as np

from sklearn.ensemble import RandomTreesEmbedding, RandomForestClassifier
from sklearn.tree import ExtraTreeRegressor


class CompletelyRandomTree:

    def __init__(self, sklearn_extra_tree, n_attributes):
        self.n_attributes = n_attributes
        self.tree = sklearn_extra_tree
        self.n_nodes = self.tree.tree_.node_count
        self.parent_ids = np.full(self.n_nodes, -1, dtype=np.int)
        self.operators = [None] * self.n_nodes
        tree = self.tree.tree_
        for i in range(self.n_nodes):
            parent_from_left = np.where(tree.children_left == i)[0]
            parent_from_right = np.where(tree.children_right == i)[0]
            if len(parent_from_left) == 1:
                self.parent_ids[i] = parent_from_left[0]
                self.operators[i] = 'le'
            elif len(parent_from_right) == 1:
                self.parent_ids[i] = parent_from_right[0]
                self.operators[i] = 'gt'
        self.leaf_to_node = np.where(tree.feature == -2)[0]
        self.node_to_leaf = {node_id: leaf_id for leaf_id, node_id in enumerate(self.leaf_to_node)}
        self.n_leafs = len(self.leaf_to_node)
    
    def encode(self, X):
        node_ids = self.tree.apply(X)
        return [self.node_to_leaf[node_id] for node_id in node_ids]
    
    def get_path_rule(self, default_path_rule, leaf_id):
        path_rule = copy.deepcopy(default_path_rule)
        tree = self.tree.tree_
        current_id = self.leaf_to_node[leaf_id]
        while current_id != 0:
            attribute = tree.feature[self.parent_ids[current_id]]
            bound = tree.threshold[self.parent_ids[current_id]]
            op = self.operators[current_id]
            path_rule.add(NodeRule(attribute, bound, op))
            current_id = self.parent_ids[current_id]
        return path_rule


class EncoderForest:

    def __init__(self, n_components):
        self.trees = list()
        self.n_attributes = 0
        self.in_size = 0
        self.out_size = n_components
        self.global_lower_bounds = None
        self.global_upper_bounds = None
        self.default_path_rule = None
        self.unsupervised = True
    
    def fit(self, X, y=None, max_depth=5):
        self.unsupervised = (y is None)
        self.n_attributes = X.shape[1]
        self.in_size = X.shape[0]
        if y is None:
            forest = RandomTreesEmbedding(self.out_size, max_depth=max_depth)
            forest.fit(X)
        else:
            forest = RandomForestClassifier(n_estimators=self.out_size, max_depth=max_depth)
            forest.fit(X, y)
        for i in range(self.out_size):
            self.trees.append(CompletelyRandomTree(forest.estimators_[i], self.n_attributes))
        
        self.global_lower_bounds = np.min(X, axis=0).astype(np.double)
        self.global_upper_bounds = np.max(X, axis=0).astype(np.double)

        self.default_path_rule = PathRule(self.n_attributes)
        self.default_path_rule.set_global_bounds(self.global_lower_bounds, self.global_upper_bounds)
    
    def encode(self, X):
        n_samples = X.shape[0]
        out_dim = len(self.trees)
        encoded = np.empty((n_samples, out_dim), dtype=np.int)
        for i in range(out_dim):
            encoded[:, i] = self.trees[i].encode(X)
        return encoded

    def compute_rule_list(self, x):
        out_dim = len(self.trees)
        rule_list = list()
        for i in range(out_dim):
            path_rule = self.trees[i].get_path_rule(self.default_path_rule, x[i])
            rule_list.append(path_rule)
        return rule_list
    
    def decode(self, x):
        rule_list = self.compute_rule_list(x)
        MCR = self.calculate_MCR(rule_list)
        return MCR.sample(sampling='mean')

    def calculate_MCR(self, path_rule_list):
        MCR = copy.deepcopy(self.default_path_rule)
        n_attributes = len(MCR)
        n_trees = len(path_rule_list)

        for i in range(len(path_rule_list)):
            path_rule = path_rule_list[i]
            print("Volume of path rule %i: %f" % (i, path_rule.compute_volume()))
            for node_rule in path_rule.node_rules:
                MCR.add(node_rule)
        print("Volume of MCR: %f" % MCR.compute_volume())
        return MCR