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


class search_tree(object):
    def __init__(self, mcst, q_id, max_depth, l_passages, p_words_list, ref_answer, vocab):
        self.tree = Tree()
        self.q_id = q_id
        self.tree.create_node(identifier='question_' + str(q_id), data=node())
        root_node = self.tree.get_node('question_' + str(q_id))
        root_node.data.num = 1.0
        self.node_map = {}
        self.l_passages = l_passages
        self.p_words_list = p_words_list
        self.ref_answer = ref_answer
        self.count = 0.0
        self.carpe_diem = mcst
        self.max_depth = max_depth
        self.vocab = vocab
        self.expand(self.tree.get_node(self.tree.root))


    def expand(self, leaf_node):
        print '---expand: '
        words_list = leaf_node.data.word
        print 'word_list:'
        print words_list
        p_word_id, p_pred = self.carpe_diem.get_policy(words_list, self.l_passages)
        print 'candidate_id: '
        print p_word_id
        for word in p_word_id:
            #print word
            #print 'self.node_map: ' + str(self.node_map)
            #print 'len of self.node_map: '+ str(len(self.node_map))
            self.node_map[' '.join(words_list + [str(word)])] = len(self.node_map)
            #print 'yi dun cao zuo'
            #print 'self.node_map: ' + str(self.node_map)
            #print 'len of self.node_map: ' + str(len(self.node_map))
            new_node = node()
            new_node.word = words_list + [str(word)]
            #print new_node.word
            new_node.p = p_pred[p_word_id.index(word)]
            new_node.Q = self.carpe_diem.value_function(words_list)[0][0]
            #print  new_node.p
            self.tree.create_node(identifier=self.node_map[' '.join(new_node.word)], data=new_node,
                                  parent=leaf_node.identifier)

    def update(self, node_list, value):
        #print '----update'
        for node_id in node_list:
            tmp_node = self.tree.get_node(node_id)
            tmp_node.data.Q = (tmp_node.data.Q * tmp_node.data.num + value) / (tmp_node.data.num + 1)
            tmp_node.data.num += 1

    def search(self, start_node_id):
        #print '----tree search'
        tmp_node = self.tree.get_node(start_node_id)
        #print tmp_node.data.num
        has_visit_num = tmp_node.data.num - 1
        self.count = has_visit_num

        if int(self.carpe_diem.search_time - has_visit_num) > 0:
            start_node_search_time = int(self.carpe_diem.search_time - has_visit_num)
            #print 'start_node_search_time: '
            #print start_node_search_time
        else:
            start_node_search_time = 0
        #print 'print str(self.l_passages): '
        #print str(self.l_passages - 1)

        for time in range(start_node_search_time):
            search_list = [start_node_id]
            tmp_node = self.tree.get_node(start_node_id)
            #print 'search time :'+ str(time)

            while not tmp_node.is_leaf():
                max_score = float("-inf")
                max_id = -1
                for child_id in tmp_node.fpointer:
                    child_node = self.tree.get_node(child_id)
                    #score = child_node.data.p
                    #print "child_node.data.p: " + str(child_node.data.p)
                    #print "tmp_node.data.num: " + str(tmp_node.data.num)
                    score = self.carpe_diem.beta * child_node.data.p * (
                    (tmp_node.data.num) ** 0.5 / (1 + child_node.data.num))

                    #print 'child_node.data.Q: '
                    #print child_node.data.Q
                    score += child_node.data.Q

                    #print 'score: '
                    #print score

                    #print '**************'

                    if score > max_score:
                        max_id = child_id
                        max_score = score
                search_list.append(max_id)
                tmp_node = self.tree.get_node(max_id)

            #query_id_mcts = self.tree.root.split('_')[1]
            #print query_id_mcts
            #print 'end'
            #print 'tmp_node.data.word'
            #print tmp_node.data.word[-1]


            #end
            #if tmp_node.data.word[-1] == str(self.l_passages-1):
            if tmp_node.data.word[-1] == str(self.l_passages - 1):
                v = 0
                pred_answer = tmp_node.data.word
                print 'pred_answer: '
                print pred_answer
                print 'listSelectedSet'
                listSelectedSet_words = []
                listSelectedSet = map(eval, pred_answer)
                print listSelectedSet
                for idx in listSelectedSet:
                    listSelectedSet_words.append(self.p_words_list[idx])
                print 'str123'
                str123 = self.vocab.recover_from_ids(listSelectedSet_words, 0)
                print str123
                pred_answers = []

                pred_answers.append({'question_id': [self.q_id],
                                     'question_type': [],
                                     'answers': [''.join(str123)],
                                     'entity_answers': [[]],
                                     'yesno_answers': []})
                if len(self.ref_answer) > 0:
                    pred_dict, ref_dict = {}, {}
                    for pred, ref in zip(pred_answers, self.ref_answer):
                        question_id = ref['question_id']
                        if len(ref['answers']) > 0:
                            pred_dict[question_id] = normalize(pred['answers'])
                            ref_dict[question_id] = normalize(ref['answers'])
                            print '========compare in tree======='
                            print pred_dict[question_id]
                            print '----------------------'
                            print ref_dict[question_id]
                    bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
                else:
                    bleu_rouge = None
                print 'bleu_rouge'
                print bleu_rouge
                v = bleu_rouge['Bleu-4']
            else:
                v = self.carpe_diem.value_function(tmp_node.data.word)[0][0]
                #print 'v: '
                #print v

            self.update(search_list, v)
            self.count += 1

            if tmp_node.is_leaf() and (self.tree.depth(tmp_node) < self.max_depth) and tmp_node.data.word[-1] != str(self.l_passages-1):
                self.expand(tmp_node)

            ###########
            '''
            if time % 100 == 0:
                tmp_policy = self.get_ppolicy(start_node_id)
                print tmp_policy.values()
                print sum(tmp_policy.values())
                print time
            '''
            #print tmp_node.data.word
            #print '------finish search '
        #print '===== finish all search ======'



    def take_action(self, start_node_id):
        #print '----take action: '
        tmp_node = self.tree.get_node(start_node_id)
        max_time = -1
        prob = {}
        for child_id in tmp_node.fpointer:
            child_node = self.tree.get_node(child_id)
            prob[child_node.data.word[-1]] = child_node.data.num / self.count

            if child_node.data.num > max_time:
                #print child_node.data.num
                #print max_time
                #print 'child_node.data.num > max_time'
                max_time = child_node.data.num
                select_word = child_node.data.word[-1]
                select_word_node_id = child_node.identifier
            #else:
                #print 'not > max time '

        #print select_word
        #print select_word_node_id
        #print '-----take action end'
        return prob, select_word, select_word_node_id

    def get_ppolicy(self, start_node_id):
        tmp_node = self.tree.get_node(start_node_id)
        max_time = -1
        prob = {}
        for child_id in tmp_node.fpointer:
            child_node = self.tree.get_node(child_id)
            if self.count == 0:
                prob[child_node.data.word[-1]] = 0.0
            else:
                prob[child_node.data.word[-1]] = child_node.data.num / self.count
        return prob


