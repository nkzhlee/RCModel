"""
Created on 2017-10-26
class: RL4SRD
@author: fengyue
"""

# !/usr/bin/python
# -*- coding:utf-8 -*-

from treelib import Tree
import copy
from utils import normalize
from utils import compute_bleu_rouge

"""
    num : number of visit time
    once_num : nothing 
    Q : value funtion calculate value of now-node 
    p : policy score of now-node 
    doc : doc episode list of now-state 
"""



class node(object):
    def __init__(self):
        self.num = 0.0
        self.Q = 0.0
        self.p = 0.0
        self.word = []


class sub_tree(object):
    def __init__(self, tree):
        self.max_depth = 50
        self.raw_tree_data = tree
        #self.mcst = tree['mcst_model']
        self.tree = Tree()
        self.q_id = tree['tree_id']
        self.tree.create_node(identifier='question_' + str(self.q_id), data=node())
        root_node = self.tree.get_node('question_' + str(self.q_id))
        root_node.data.num = 1.0
        self.node_map = {}
        self.ref_answer = tree['ref_answer']
        self.count = 0.0
        #self.expand(self.tree.get_node(self.tree.root))

    def get_raw_tree_data(self):
        return self.raw_tree_data
