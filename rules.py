# -*- coding: utf-8 -*-
# rules.py
# author : Antoine Passemiers

import copy
import numpy as np


class NodeRule:

    LT = 0 # 'less than'
    LE = 1 # 'less or equal to'

    def __init__(self, attribute, threshold, op):
        # [lower_bound] [op_left] value [op_right] [upper_bound]
        assert(op in ['gt', 'le'])
        self.attribute = attribute
        if op == 'le':
            self.lower_bound = -np.inf
            self.upper_bound = threshold
            self.op_left = NodeRule.LT
            self.op_right = NodeRule.LE
        else:
            self.lower_bound = threshold
            self.upper_bound = np.inf
            self.op_left = NodeRule.LT
            self.op_right = NodeRule.LT
    
    def __and__(self, other):
        new_rule = copy.copy(self)
        if self.lower_bound > other.lower_bound:
            new_rule.lower_bound = self.lower_bound
            new_rule.op_left = self.op_left
        elif self.lower_bound < other.lower_bound:
            new_rule.lower_bound = other.lower_bound
            new_rule.op_left = other.op_left
        else:
            new_rule.lower_bound = self.lower_bound
            if NodeRule.LT in [self.op_left, other.op_left]:
                new_rule.op_left = NodeRule.LT
            else:
                new_rule.op_left = NodeRule.LE
        if self.upper_bound < other.upper_bound:
            new_rule.upper_bound = self.upper_bound
            new_rule.op_right = self.op_right
        elif self.upper_bound > other.upper_bound:
            new_rule.upper_bound = other.upper_bound
            new_rule.op_right = other.op_right
        else:
            new_rule.upper_bound = self.upper_bound
            if NodeRule.LT in [self.op_right, other.op_right]:
                new_rule.op_right = NodeRule.LT
            else:
                new_rule.op_right = NodeRule.LE
        return new_rule

    def __str__(self):
        s = str(self.lower_bound) + ' '
        s += '<=' if self.op_left == NodeRule.LE else '<'
        s += ' x_%i ' % self.attribute
        s += '<=' if self.op_right == NodeRule.LE else '<'
        s += ' ' + str(self.upper_bound)
        return s
    
    def __repr__(self):
        return self.__str__()


class PathRule:
    
    def __init__(self, n_attributes):
        self.n_attributes = n_attributes
        self.node_rules = [NodeRule(i, -np.inf, 'gt') for i in range(n_attributes)]
    
    def set_global_bounds(self, lower_bounds, upper_bounds):
        for i in range(self.n_attributes):
            node_rule = NodeRule(i, -np.inf, 'gt')
            node_rule.op_left = NodeRule.LE
            node_rule.op_right = NodeRule.LE
            node_rule.lower_bound = lower_bounds[i]
            node_rule.upper_bound = upper_bounds[i]
            self.add(node_rule)
    
    def add(self, node_rule):
        assert(isinstance(node_rule, NodeRule))
        j = node_rule.attribute
        self.node_rules[j] = self.node_rules[j].__and__(node_rule)
    
    def sample(self, sampling='mean'):
        sample = np.empty(self.n_attributes)
        for i, rule in enumerate(self):
            if sampling == 'mean':
                sample[i] = (rule.lower_bound + rule.upper_bound) / 2.
            else:
                sample[i] = np.random.uniform(rule.lower_bound, rule.upper_bound)
        return sample
    
    def compute_volume(self):
        log_volume = 1
        for i in range(self.n_attributes):
            node_rule = self.node_rules[i]
            log_volume += (node_rule.upper_bound - node_rule.lower_bound)
            assert(node_rule.upper_bound >= node_rule.lower_bound)
        return log_volume
    
    def __len__(self):
        return len(self.node_rules)
    
    def __iter__(self):
        self.current_attribute = 0
        return self
    
    def __next__(self):
        self.current_attribute += 1
        try:
            result = self.node_rules[self.current_attribute-1]
        except IndexError:
            raise StopIteration
        return result
    
    def __str__(self):
        return str(self.node_rules)
    
    def __repr__(self):
        return self.__str__()
