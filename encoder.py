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
    
    def encode(self, X):
        return self.tree.apply(X)
    
    def get_path_rule(self, default_path_rule, leaf_id):
        path_rule = copy.deepcopy(default_path_rule)
        tree = self.tree.tree_
        current_id = leaf_id
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
        
        self.global_lower_bounds = np.min(X, axis=0)
        self.global_upper_bounds = np.max(X, axis=1)

        self.default_path_rule = PathRule(self.n_attributes)
        for i in range(self.n_attributes):
            node_rule = NodeRule(i, -np.inf, 'gt')
            node_rule.op_left = NodeRule.LE
            node_rule.op_right = NodeRule.LE
            node_rule.lower_bound = self.global_lower_bounds[i]
            node_rule.upper_bound = self.global_upper_bounds[i]
            self.default_path_rule.add(node_rule)
    
    def encode(self, X):
        n_samples = X.shape[0]
        out_dim = len(self.trees)
        encoded = np.empty((n_samples, out_dim), dtype=np.int)
        for i in range(out_dim):
            encoded[:, i] = self.trees[i].encode(X)
        return encoded
    
    def decode(self, x):
        out_dim = len(self.trees)
        rule_list = list()
        for i in range(out_dim):
            path_rule = self.trees[i].get_path_rule(self.default_path_rule, x[i])
            rule_list.append(path_rule)
        
        MCR = self.calculate_MCR(rule_list)
        return MCR.sample()

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