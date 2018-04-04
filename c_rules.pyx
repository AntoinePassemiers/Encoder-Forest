# -*- coding: utf-8 -*-
# rules.pyx
# author : Antoine Passemiers
# distutils: language=c
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=False

import numpy as np
cimport numpy as cnp
cnp.import_array()

cimport libc.math
from libc.stdlib cimport *
from libc.string cimport memcpy


cdef enum RuleRelation:
    LT, LE, GT


cdef struct NodeRule:
    int attribute
    double lower_bound
    double upper_bound
    RuleRelation op_left
    RuleRelation op_right


cdef void init_node_rule(NodeRule* node_rule, int attribute, double threshold, RuleRelation op) nogil:
    node_rule.attribute = attribute
    if op == LE:
        node_rule.lower_bound = -libc.math.HUGE_VAL
        node_rule.upper_bound = threshold
        node_rule.op_left = LT
        node_rule.op_right = LE
    else:
        node_rule.lower_bound = threshold
        node_rule.upper_bound = libc.math.HUGE_VAL
        node_rule.op_left = LT
        node_rule.op_right = LT


cdef NodeRule apply_and_rule(NodeRule rule_a, NodeRule rule_b) nogil:
    cdef NodeRule new_rule
    if rule_a.lower_bound > rule_b.lower_bound:
        new_rule.lower_bound = rule_a.lower_bound
        new_rule.op_left = rule_a.op_left
    elif rule_a.lower_bound < rule_b.lower_bound:
        new_rule.lower_bound = rule_b.lower_bound
        new_rule.op_left = rule_b.op_left
    else:
        new_rule.lower_bound = rule_a.lower_bound
        if (rule_a.op_left == LT) or (rule_b.op_left == LT):
            new_rule.op_left = LT
        else:
            new_rule.op_left = LE
    if rule_a.upper_bound < rule_b.upper_bound:
        new_rule.upper_bound = rule_a.upper_bound
        new_rule.op_right = rule_a.op_right
    elif rule_a.upper_bound > rule_b.upper_bound:
        new_rule.upper_bound = rule_b.upper_bound
        new_rule.op_right = rule_b.op_right
    else:
        new_rule.upper_bound = rule_a.upper_bound
        if (rule_a.op_right == LT) or (rule_b.op_right == LT):
            new_rule.op_right = LT
        else:
            new_rule.op_right = LE
    return new_rule


cdef void add_rule(NodeRule* node_rules, NodeRule node_rule) nogil:
    cdef int j = node_rule.attribute
    node_rules[j] = apply_and_rule(node_rules[j], node_rule)


cdef class PathRule:

    cdef int n_attributes
    cdef NodeRule* node_rules

    def __cinit__(self, int n_attributes):
        self.n_attributes = n_attributes
        cdef int attribute
        self.node_rules = <NodeRule*>malloc(n_attributes * sizeof(NodeRule))
        with nogil:
            for attribute in range(self.n_attributes):
                init_node_rule(&self.node_rules[attribute], attribute, -libc.math.HUGE_VAL, GT)
    
    def set_global_bounds(self, cnp.double_t[:] lower_bounds, cnp.double_t[:] upper_bounds):
        cdef int attribute
        with nogil:
            for attribute in range(self.n_attributes):
                self.node_rules[attribute].op_left = LE
                self.node_rules[attribute].op_right = LE
                self.node_rules[attribute].lower_bound = lower_bounds[attribute]
                self.node_rules[attribute].upper_bound = upper_bounds[attribute]
    
    def add(self, NodeRule node_rule):
        add_rule(self.node_rules, node_rule)

    def sample(self):
        cdef cnp.float_t[:] sample = np.empty(self.n_attributes, dtype=np.float)
        cdef int i
        with nogil:
            for i in range(self.n_attributes):
                sample[i] = (self.node_rules[i].lower_bound + self.node_rules[i].upper_bound) / 2.
        return np.asarray(sample)

    def compute_volume(self):
        cdef double log_volume = 1.
        cdef int i
        with nogil:
            for i in range(self.n_attributes):
                log_volume += (self.node_rules[i].upper_bound - self.node_rules[i].lower_bound)
        return log_volume

    def __dealloc__(self):
        free(self.node_rules)

    def __copy__(self):
        new_path_rule = PathRule(self.n_attributes)
        new_path_rule.node_rules = self.node_rules
        return new_path_rule
    
    def __deepcopy__(self):
        new_path_rule = PathRule(self.n_attributes)
        memcpy(new_path_rule.node_rules, self.node_rules, self.n_attributes * sizeof(NodeRule))
        return new_path_rule
    
    def __len__(self):
        return self.n_attributes
    
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

    def node_rule_to_str(self, NodeRule node_rule):
        s = str(node_rule.lower_bound) + ' '
        s += '<=' if node_rule.op_left == LE else '<'
        s += ' x_%i ' % node_rule.attribute
        s += '<=' if node_rule.op_right == LE else '<'
        s += ' ' + str(node_rule.upper_bound)
        return s
    
    def __str__(self):
        s = str()
        for i in range(self.n_attributes):
            s += self.node_rule_to_str(self.node_rules[i]) + ", "
        return s
    
    def __repr__(self):
        return self.__str__()
