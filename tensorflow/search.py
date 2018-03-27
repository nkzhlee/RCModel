import os
import time
import logging
import json
from mctree import search_tree
import numpy as np
import tensorflow as tf
from utils import compute_bleu_rouge
from utils import normalize
from layers.basic_rnn import rnn
from layers.match_layer import MatchLSTMLayer
from layers.match_layer import AttentionFlowMatchLayer
from layers.pointer_net import PointerNetDecoder

class SearchTree(object):
    """
    Implements the main reading comprehension model.
    
    python -u run.py --train --algo MCST --epochs 2 --batch_size 1 --max_p_len 10000 --hidden_size 150  --train_files ../data/demo/trainset/test10 --dev_files  ../data/demo/devset/test5 --test_files ../data/demo/test/search.test.json
    
    python -u run.py --train --algo BIDAF --epochs 2 --batch_size 1 --max_p_len 10000 --hidden_size 150  --train_files ../data/demo/trainset/test10 --dev_files  ../data/demo/devset/test20 --test_files ../data/demo/test/search.test.json
    
    
    nohup python -u run.py --train --algo MCST --epochs 20 --batch_size 1 --gpu 0 --max_p_len 10000 --hidden_size 150  --train_files ../data/demo/trainset/search.train.json --dev_files  ../data/demo/devset/search.dev.json --test_files ../data/demo/test/search.test.json >lalallala.txt 2>&1 &
    
    nohup python -u run.py --train --algo MCST --epochs 20 --batch_size 1 --gpu 2 --max_p_len 10000 --hidden_size 150  --train_files ../data/demo/trainset/search.train.json --dev_files  ../data/demo/devset/search.dev.json --test_files ../data/demo/test/search.test.json >laa_10_500.txt 2>&1 &
    
    nohup python -u run.py --train --algo MCST --epochs 10 --batch_size 1 --gpu 3 --max_p_len 10000 --hidden_size 150  --train_files ../data/demo/trainset/search.train.json --dev_files  ../data/demo/devset/search.dev.json --test_files ../data/demo/test/search.test.json >dababao.txt 2>&1 &
    
    """

    def __init__(self, tfg, data, max_a_len, max_search_time, beta, dropout_keep_prob):
        self.tfg = tfg
        self.data = data
        self.beta = beta
        self.max_a_len = max_a_len
        self.max_search_time = max_search_time
        self.dropout_keep_prob = dropout_keep_prob
        self.l_passage = 0
    #start search
    def one_train(self):
        #print 'new search def'
        p_data = []
        listSelectedSet = []
        # print self.data
        # #init feed_dict
        # print ('passage_token_ids',np.shape([self.data['passage_token_ids']]))
        # print ('question_token_ids', np.shape([self.data['question_token_ids']]))
        # print ('p_length', np.shape([self.data['p_length']]))
        # print ('q_length', np.shape([self.data['q_length']]))

        batch_start_time = time.time()
        self.feed_dict = {self.tfg.p: [self.data['passage_token_ids']],
                          self.tfg.q: [self.data['question_token_ids']],
                          self.tfg.p_length: [self.data['p_length']],
                          self.tfg.q_length: [self.data['q_length']],
                          self.dropout_keep_prob: self.dropout_keep_prob}
        #init answers
        self.l_passage = len(self.data['passage_token_ids'])

        pred_answers, ref_answers = [], []
        ref_answers.append({'question_id': self.data['question_id'],
                                'question_type': self.data['question_type'],
                                    'answers': self.data['ref_answers'],
                                    'entity_answers': [[]],
                                    'yesno_answers': []})
        #init tree
        self.tfg.set_feed_dict([self.data['passage_token_ids']],
                           [self.data['question_token_ids']],
                           [self.data['p_length']],
                           [self.data['q_length']],
                           self.dropout_keep_prob)
        # shape_a, shape_b = self.tfg.run_session_shape()
        # print ('shape_a', shape_a)
        # print ('shape_b', shape_b)
        start_node = 'question_' + str(self.data['question_id'])
        mcts_tree = search_tree(self.tfg, self.data['question_id'], self.data['passage_token_ids'], self.max_a_len,
                                self.max_search_time, self.beta, self.l_passage, ref_answers, self.tfg.vocab)
        for t in range(self.max_a_len):
            #print ('Answer_len', t)
            mcts_tree.search(start_node)
            tmp_policy = mcts_tree.get_ppolicy(start_node)
            # # print 'tmp_policy.values(): '
            # # print tmp_policy.values()
            # print 'sum(tmp_policy.values()): '
            # print sum(tmp_policy.values())
            prob, select_doc_id, start_node = mcts_tree.take_action(start_node)
            p_data.append(prob)
            listSelectedSet.append(select_doc_id)
            if select_doc_id == str(self.l_passage-1):
                #print 'break!!!!!!!!!!!'
                break
        #print ('listSelectedSet', listSelectedSet)
        listSelectedSet_words = []
        listSelectedSet = map(eval, listSelectedSet)
        for idx in listSelectedSet:
            listSelectedSet_words.append(self.data['passage_token_ids'][idx])
            # print 'listSelectedSet:'
            # print listSelectedSet
            # print 'listSelectedSet_words: '
            # print listSelectedSet_words
        strr123 = self.tfg.vocab.recover_from_ids(listSelectedSet_words, 0)
            # print strr123
        pred_answers.append({'question_id': self.data['question_id'],
                                'question_type': self.data['question_type'],
                                 'answers': [''.join(strr123)],
                                 'entity_answers': [[]],
                                 'yesno_answers': []})

        # print 'ref_answers: '
        # print ref_answers
        # print 'pred_answer: '
        # print pred_answers

        if len(ref_answers) > 0:
            pred_dict, ref_dict = {}, {}
            for pred, ref in zip(pred_answers, ref_answers):
                question_id = ref['question_id']
                if len(ref['answers']) > 0:
                    pred_dict[question_id] = normalize(pred['answers'])
                    ref_dict[question_id] = normalize(ref['answers'])
                    # print '========compare======='
                    # print pred_dict[question_id]
                    # print '----------------------'
                    # print ref_dict[question_id]
            # print '========compare 2======='
            # print pred_dict
            # print '----------------------'
            # print ref_dict
            bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
        else:
            bleu_rouge = None
        value_with_mcts = bleu_rouge
        #print 'bleu_rouge(value_with_mcts): '
        #print value_with_mcts
        # now use Bleu-4 , Rouge-L
        input_v = value_with_mcts['Bleu-4']
        #print self.data
        total_loss = 0
        num_loss = 0
        for prob_id, prob_data in enumerate(p_data):
            # print 'p_data: '
            # print prob_id
            # print prob_data
            num_loss += 1
            c = []
            policy = []
            for prob_key, prob_value in prob_data.items():
                c.append(prob_key)
                policy.append(prob_value)
            # print 'policy: '
            # print [policy]
            # print 'value: '
            # print [value_with_mcts]
            # print 'candidate: '
            # print c
            if prob_id == 0:
                loss = self.tfg.cal_first_loss(policy, input_v)
            else:
                loss = self.tfg.cal_loss(policy,input_v,listSelectedSet[:prob_id],c)
            total_loss += loss
            print ('loss', loss)
        #print ('&&&&&&&&&&&&&&& 1 batch train time = %3.2f s &&&&&&&&&&&&' % (time.time() - batch_start_time))
        return 1.0 * total_loss / num_loss

    def one_evaluate(self):
        listSelectedSet = []
        self.feed_dict = {self.tfg.p: [self.data['passage_token_ids']],
                          self.tfg.q: [self.data['question_token_ids']],
                          self.tfg.p_length: [self.data['p_length']],
                          self.tfg.q_length: [self.data['q_length']],
                          self.dropout_keep_prob: self.dropout_keep_prob}
        #init answers
        self.l_passage = len(self.data['passage_token_ids'])

        pred_answers, ref_answers = [], []
        ref_answers.append({'question_id': self.data['question_id'],
                                'question_type': self.data['question_type'],
                                    'answers': self.data['ref_answers'],
                                    'entity_answers': [[]],
                                    'yesno_answers': []})
        #init tree
        self.tfg.set_feed_dict([self.data['passage_token_ids']],
                           [self.data['question_token_ids']],
                           [self.data['p_length']],
                           [self.data['q_length']],
                           self.dropout_keep_prob)
        # shape_a, shape_b = self.tfg.run_session_shape()
        # print ('shape_a', shape_a)
        # print ('shape_b', shape_b)
        start_node = 'question_' + str(self.data['question_id'])
        mcts_tree = search_tree(self.tfg, self.data['question_id'], self.data['passage_token_ids'], self.max_a_len,
                                self.max_search_time, self.beta, self.l_passage, ref_answers, self.tfg.vocab)
        for t in range(self.max_a_len):
            #print ('Answer_len', t)
            mcts_tree.search_eval(start_node)
            tmp_policy = mcts_tree.get_ppolicy(start_node)
            # # print 'tmp_policy.values(): '
            # # print tmp_policy.values()
            prob, select_doc_id, start_node = mcts_tree.take_action(start_node)
            listSelectedSet.append(select_doc_id)
            if select_doc_id == str(self.l_passage-1):
                #print 'break!!!!!!!!!!!'
                break
        #print ('listSelectedSet', listSelectedSet)
        listSelectedSet_words = []
        listSelectedSet = map(eval, listSelectedSet)
        for idx in listSelectedSet:
            listSelectedSet_words.append(self.data['passage_token_ids'][idx])
        # print 'listSelectedSet:'
        # print listSelectedSet
        # print 'listSelectedSet_words: '
        # print listSelectedSet_words
        strr123 = self.tfg.vocab.recover_from_ids(listSelectedSet_words, 0)
        #print strr123
        pred_answer ={'question_id': self.data['question_id'],
                                'question_type': self.data['question_type'],
                                 'answers': [''.join(strr123)],
                                 'entity_answers': [[]],
                                 'yesno_answers': []}

        # print pred_answer
        # print '++++++++++++++++ end evaluate +++++++++++++++++'
        return pred_answer