# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 lizhaohui.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module implements the reading comprehension models based on:
Reinforcement Learning and Monte-Carlo Tree Search
Note that we use Pointer Network for the decoding stage of both models.
"""

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
from pmctree import PSCHTree

class MCSTmodel(object):
    """
    Implements the main reading comprehension model.
    """

    def __init__(self, vocab, args):

        # logging
        self.args = args
        self.logger = logging.getLogger("brc")

        # basic config
        self.algo = args.algo
        self.hidden_size = args.hidden_size
        self.optim_type = args.optim
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.use_dropout = args.dropout_keep_prob < 1

        # length limit
        self.max_p_num = args.max_p_num
        self.max_p_len = args.max_p_len
        self.max_q_len = args.max_q_len
        #self.max_a_len = args.max_a_len
        self.max_a_len = 20
        #test paras
        self.search_time = 3000
        self.beta = 100.0

        # the vocab
        self.vocab = vocab
        #self._build_graph()



    def _build_graph(self):
        """
        Builds the computation graph with Tensorflow
        """
        # session info
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        start_t = time.time()
        self._setup_placeholders()
        self._embed()
        self._encode()
        self._initstate()
        self._action_frist()
        self._action()
        self._compute_loss()
        #param_num = sum([np.prod(self.sess.run(tf.shape(v))) for v in self.all_params])
        #self.logger.info('There are {} parameters in the model'.format(param_num))
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))

    def _setup_placeholders(self):
        """
        Placeholders
        """
        self.p = tf.placeholder(tf.int32, [None, None])
        self.q = tf.placeholder(tf.int32, [None, None])
        self.p_length = tf.placeholder(tf.int32, [None])
        self.q_length = tf.placeholder(tf.int32, [None])
        self.start_label = tf.placeholder(tf.int32, [None])
        self.end_label = tf.placeholder(tf.int32, [None])
        self.dropout_keep_prob = tf.placeholder(tf.float32)

        #test
        self.p_words_id = tf.placeholder(tf.int32, [None])
        self.candidate_id = tf.placeholder(tf.int32, [None])
        #self.words = tf.placeholder(tf.float32, [None, None])
        self.selected_id_list = tf.placeholder(tf.int32, [None])
        self.policy = tf.placeholder(tf.float32, [1, None])  # policy
        self.v = tf.placeholder(tf.float32, [1, 1])  # value

    def _embed(self):
        """
        The embedding layer, question and passage share embeddings
        """
        #with tf.device('/cpu:0'), tf.variable_scope('word_embedding'):
        with tf.variable_scope('word_embedding'):
            self.word_embeddings = tf.get_variable(
                'word_embeddings',
                shape=(self.vocab.size(), self.vocab.embed_dim),
                initializer=tf.constant_initializer(self.vocab.embeddings),
                trainable=True
            )
            self.p_emb = tf.nn.embedding_lookup(self.word_embeddings, self.p)
            self.q_emb = tf.nn.embedding_lookup(self.word_embeddings, self.q)

    def _encode(self):
        """
        Employs two Bi-LSTMs to encode passage and question separately
        """
        with tf.variable_scope('passage_encoding'):
            self.p_encodes, _ = rnn('bi-lstm', self.p_emb, self.p_length, self.hidden_size)
        with tf.variable_scope('question_encoding'):
            _, self.sep_q_encodes= rnn('bi-lstm', self.q_emb, self.q_length, self.hidden_size)
        if self.use_dropout:
            self.p_encodes = tf.nn.dropout(self.p_encodes, self.dropout_keep_prob)
            self.sep_q_encodes = tf.nn.dropout(self.sep_q_encodes, self.dropout_keep_prob)
    def _initstate(self):
        self.V = tf.Variable(tf.random_uniform([self.hidden_size*2, self.hidden_size * 2], -1. / self.hidden_size,1. / self.hidden_size))
        self.W = tf.Variable(tf.random_uniform([self.hidden_size * 2, 1], -1. / self.hidden_size, 1. / self.hidden_size))

        self.W_b = tf.Variable(tf.random_uniform([1, 1], -1. / self.hidden_size, 1. / self.hidden_size))

        self.V_c = tf.Variable(tf.random_uniform([self.hidden_size*2, self.hidden_size], -1. / self.hidden_size, 1. / self.hidden_size))
        self.V_h = tf.Variable(tf.random_uniform([self.hidden_size*2, self.hidden_size], -1. / self.hidden_size, 1. / self.hidden_size))

        self.q_state_c = tf.sigmoid(tf.matmul(self.sep_q_encodes, self.V_c))
        self.q_state_h = tf.sigmoid(tf.matmul(self.sep_q_encodes, self.V_h))
        self.q_state = tf.concat([self.q_state_c, self.q_state_h], 1)

        self.words = tf.reshape(self.p_encodes,[-1,self.hidden_size*2])

        self.words_list = tf.gather(self.words, self.p_words_id) # all words in a question doc
    def _action_frist(self):
        """
        select first word
        """
        #self.candidate = tf.reshape(self.p_emb,[-1,self.hidden_size*2])
        self.logits_first = tf.reshape(tf.matmul(tf.matmul(self.words_list, self.V), tf.transpose(self.q_state)), [-1])
        self.prob_first = tf.nn.softmax(self.logits_first)
        self.prob_id_first = tf.argmax(self.prob_first)
        self.value_first = tf.sigmoid(tf.reshape(tf.matmul(self.q_state, self.W), [1, 1]) + self.W_b)  # [1,1]



    def _action(self):
        """
        Employs Bi-LSTM again to fuse the context information after match layer
        """
        self.candidate = tf.gather(self.words_list, self.candidate_id)
        self.selected_list = tf.gather(self.words_list, self.selected_id_list)
        self.input = tf.reshape(self.selected_list, [1, -1, self.hidden_size*2])
        rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_size, state_is_tuple=False)
        _, self.states = tf.nn.dynamic_rnn(rnn_cell, self.input, initial_state=self.q_state, dtype=tf.float32)  # [1, dim]
        self.logits = tf.reshape(tf.matmul(tf.matmul(self.candidate, self.V), tf.transpose(self.states)), [-1])

        self.prob = tf.nn.softmax(self.logits)
        self.prob_id = tf.argmax(self.prob)
        self.value = tf.sigmoid(tf.reshape(tf.matmul(self.states, self.W), [1, 1]) + self.W_b)  # [1,1]



    def value_function(self, words_list):
        words_list = map(eval, words_list)
        #print words_list
        if len(words_list) == 0:
            value_p = self.sess.run(self.value_first, feed_dict=self.feed_dict)
        else:
            feed_dict = dict({self.selected_id_list: words_list}.items() + self.feed_dict.items())
            value_p = self.sess.run(self.value, feed_dict=feed_dict)
        return value_p

    def get_policy(self, words_list, l_passages):

        max_id = float('-inf')
        policy_c_id = []
        words_list = map(eval, words_list)
        for can in words_list:
            max_id = max(can,max_id)
        for idx in range(l_passages):
            if idx > max_id:
                policy_c_id.append(idx)
        if len(words_list) == 0:
            c_pred = self.sess.run(self.prob_first, feed_dict=self.feed_dict)
        else:
            feed_dict = dict({self.selected_id_list: words_list, self.candidate_id: policy_c_id}.items() + self.feed_dict.items())
            c_pred = self.sess.run(self.prob, feed_dict=feed_dict)

        return policy_c_id, c_pred
    def _decode(self):
        """
        Employs Pointer Network to get the the probs of each position
        to be the start or end of the predicted answer.
        Note that we concat the fuse_p_encodes for the passages in the same document.
        And since the encodes of queries in the same document is same, we select the first one.
        """
        with tf.variable_scope('same_question_concat'):
            batch_size = tf.shape(self.start_label)[0]
            concat_passage_encodes = tf.reshape(
                self.fuse_p_encodes,
                [batch_size, -1, 2 * self.hidden_size]
            )
            no_dup_question_encodes = tf.reshape(
                self.sep_q_encodes,
                [batch_size, -1, tf.shape(self.sep_q_encodes)[1], 2 * self.hidden_size]
            )[0:, 0, 0:, 0:]
        decoder = PointerNetDecoder(self.hidden_size)
        self.start_probs, self.end_probs = decoder.decode(concat_passage_encodes,
                                                          no_dup_question_encodes)

    def _compute_loss(self):
        """
        The loss function
        """
        self.loss_first = tf.contrib.losses.mean_squared_error(self.v, self.value_first) - tf.matmul(self.policy, tf.reshape(
            tf.log(tf.clip_by_value(self.prob_first, 1e-30, 1.0)), [-1, 1]))
        self.optimizer_first = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss_first)
        self.loss = tf.contrib.losses.mean_squared_error(self.v, self.value) - tf.matmul(self.policy, tf.reshape(
            tf.log(tf.clip_by_value(self.prob, 1e-30, 1.0)), [-1, 1]))
        self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)
        self.all_params = tf.trainable_variables()

    def _create_train_op(self):
        """
        Selects the training algorithm and creates a train operation with it
        """
        if self.optim_type == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.optim_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.optim_type == 'rprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.optim_type == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            raise NotImplementedError('Unsupported optimizer: {}'.format(self.optim_type))
        self.train_op = self.optimizer.minimize(self.loss)

    def _train_epoch_new(self, pmct, train_batches, batch_size, dropout_keep_prob):
        """
               Trains the model for a single epoch.
               Args:
                   train_batches: iterable batch data for training
                   dropout_keep_prob: float value indicating dropout keep probability
               """


        for bitx, batch in enumerate(train_batches, 1):
            print '------ Batch Question: ' + str(bitx)
            '''
            feed_dict = {self.p: batch['passage_token_ids'],
                              self.q: [batch['question_token_ids']],
                              self.p_length: batch['passage_length'],
                              self.q_length: [batch['question_length']],
                              self.dropout_keep_prob: dropout_keep_prob}
            '''
            pred_answers = {}
            #print str(ref_answers)
            listSelectedSet = []
            p_data = []
            tree_batch = {
                'tree_ids': batch['question_ids'],
                'question_type': batch['question_types'],
                'root_tokens': batch['question_token_ids'],
                'q_length': batch['question_length'],
                'candidates': batch['passage_token_ids'],
                'p_length': batch['passage_length'],
                'ref_answers': batch['ref_answers'],
                'mcst_model': self
            }
            feed_dict = {}
            pmct.feed_in_batch(tree_batch, 3, feed_dict)
            loss = pmct.tree_search()

        return loss

    def _train_epoch(self, train_batches, dropout_keep_prob):
        """
        Trains the model for a single epoch.
        Args:
            train_batches: iterable batch data for training
            dropout_keep_prob: float value indicating dropout keep probability
        """

        total_num, total_loss = 0, 0
        log_every_n_batch, n_batch_loss = 3, 0

        for bitx, batch in enumerate(train_batches, 1):
            print '------ Batch Question: ' + str(bitx)
            #print 'each passage len: '
            #print batch['padded_p_len']
            p_words_id = [] #all words id
            p_words_list = [] #all words except padding
            p_words_list_all = []
            l_passages = 1 # include end_pad
            n = 0
            for l in batch['passage_length']:
                l_passages += l
                temp_id = [i + n * (int(batch['padded_p_len'])) for i in range(l)]
                #print temp
                temp_w = batch['passage_token_ids'][n][:l]
                temp_all = batch['passage_token_ids'][n]
                n += 1
                p_words_id += temp_id
                p_words_list += temp_w
                p_words_list_all += temp_all
            p_words_list.append(0)
            self.end_pad = []
            self.end_pad.append(p_words_id[-1] + 1)
            p_words_id.append(p_words_id[-1] + 1)
            #print 'end_pad: '
            #print self.end_pad
            #print p_words_list
            #print p_words_id
            #print len(p_words_list)
            #print p_words_list_all
            #print len(p_words_list_all)
            self.max_a_len = min(self.max_a_len, l_passages)

            self.feed_dict = {self.p: batch['passage_token_ids'],
                         self.q: [batch['question_token_ids'][0]],
                         self.p_length: batch['passage_length'],
                         self.q_length: [batch['question_length'][0]],
                         self.start_label: batch['start_id'],
                         self.end_label: batch['end_id'],
                         self.p_words_id: p_words_id,
                         self.dropout_keep_prob: dropout_keep_prob}

            #print "question_length: " + str(batch['question_length'])
            #print "passage_length: " + str(batch['passage_length'])

            pred_answers, ref_answers = [], []

            for sample in batch['raw_data']:
                if 'answers' in sample:
                    ref_answers.append({'question_id': sample['question_id'],
                                         'question_type': sample['question_type'],
                                         'answers': sample['answers'],
                                         'entity_answers': [[]],
                                         'yesno_answers': []})
                #print 'answers: '
                #print str(sample['answers'])

            #print 'ref_answers: '
            #print ref_answers

            listSelectedSet = []
            p_data = []
            start_node = 'question_'+ str(batch['question_ids'][0])
            mcts_tree = search_tree(self, batch['question_ids'][0], self.max_a_len, l_passages, p_words_list, ref_answers, self.vocab)

            #for t in xrange(3):
            for t in xrange(self.max_a_len):

                #print '-------------'+str(t)+'------------'
                mcts_tree.search(start_node)
                tmp_policy = mcts_tree.get_ppolicy(start_node)
                #print 'tmp_policy.values(): '
                #print tmp_policy.values()
                #print 'sum(tmp_policy.values()): '
                #print sum(tmp_policy.values())

                prob, select_doc_id, start_node = mcts_tree.take_action(start_node)
                p_data.append(prob)
                listSelectedSet.append(select_doc_id)
                if select_doc_id in self.end_pad:
                    print 'break!!!!!!!!!!!'
                    break

            listSelectedSet_words = []
            listSelectedSet = map(eval, listSelectedSet)
            for idx in listSelectedSet:
                listSelectedSet_words.append(p_words_list[idx])
            #print 'listSelectedSet:'
            #print listSelectedSet
            #print 'listSelectedSet_words: '
           #print listSelectedSet_words
            for sample in batch['raw_data']:
                #print 'str：'
                strr123 = self.vocab.recover_from_ids(listSelectedSet_words, 0)
                #print strr123
                pred_answers.append({'question_id': sample['question_id'],
                                 'question_type': sample['question_type'],
                                  'answers': [''.join(strr123)],
                                    'entity_answers': [[]],
                                    'yesno_answers': []})

            #print 'pred_answer: '
            #print pred_answers

            if len(ref_answers) > 0:
                pred_dict, ref_dict = {}, {}
                for pred, ref in zip(pred_answers, ref_answers):
                    question_id = ref['question_id']
                    if len(ref['answers']) > 0:
                        pred_dict[question_id] = normalize(pred['answers'])
                        ref_dict[question_id] = normalize(ref['answers'])
                        #print '========compare======='
                        #print pred_dict[question_id]
                        #print '----------------------'
                        #print ref_dict[question_id]
                #print '========compare 2======='
                #print pred_dict
                #print '----------------------'
                #print ref_dict
                bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
            else:
                bleu_rouge = None
            value_with_mcts = bleu_rouge
            print 'bleu_rouge(value_with_mcts): '
            print value_with_mcts
            # now use Bleu-4 , Rouge-L
            input_v = value_with_mcts['Bleu-4']
            for prob_id, prob_data in enumerate(p_data):
                #print 'p_data: '
                #print prob_id
                #print prob_data
                c = []
                policy = []
                for prob_key, prob_value in prob_data.items():
                    c.append(prob_key)
                    policy.append(prob_value)
                #print 'policy: '
                #print [policy]
                #print 'value: '
                #print [value_with_mcts]
                #print 'candidate: '
                #print c
                if prob_id == 0:
                    feed_dict = dict(self.feed_dict.items() + {self.policy: [policy], self.v: [[input_v]]}.items())
                    _, loss = self.sess.run([self.optimizer_first,self.loss_first], feed_dict=feed_dict)
                else:
                    feed_dict = dict(self.feed_dict.items() + {self.selected_id_list: listSelectedSet[:prob_id], self.candidate_id: c, self.policy: [policy],
                                             self.v: [[input_v]]}.items())
                    _, loss = self.sess.run([self.optimizer,self.loss], feed_dict=feed_dict)

                total_loss += loss * len(batch['raw_data'])
                total_num += len(batch['raw_data'])
                n_batch_loss += loss
                if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                    self.logger.info('Average loss from batch {} to {} is {}'.format(
                        bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch))
                n_batch_loss = 0

            return 1.0 * total_loss / total_num

    def train(self, data, epochs, batch_size, save_dir, save_prefix,
              dropout_keep_prob=1.0, evaluate=True):
        """
        Train the model with data
        Args:
            data: the BRCDataset class implemented in dataset.py
            epochs: number of training epochs
            batch_size:
            save_dir: the directory to save the model
            save_prefix: the prefix indicating the model type
            dropout_keep_prob: float value indicating dropout keep probability
            evaluate: whether to evaluate the model on test set after each epoch
        """
        pad_id = self.vocab.get_id(self.vocab.pad_token)
        print 'pad_id is '
        print pad_id
        max_bleu_4 = 0
        for epoch in range(1, epochs + 1):
            self.logger.info('Training the model for epoch {}'.format(epoch))
            epoch_start_time = time.time()
            train_batches = data.gen_batches('train', 3, pad_id, shuffle=True)
            #mctree = MCtree(train_batches)
            #mctree.search()
            pmct = PSCHTree(self.args,self.vocab)
            result = self._train_epoch_new(pmct, train_batches, batch_size, dropout_keep_prob)



            epoch_end_time = time.time()
            self.logger.info('Train time for epoch {} is {} min'.format(epoch, str((epoch_end_time - epoch_start_time)/60)))
            #train_batches = data.gen_mini_batches('train', batch_size, pad_id, shuffle=True)
            #result = self._train_epoch(train_batches, dropout_keep_prob)

            #self.logger.info('Average train loss for epoch {} is {}'.format(epoch, result))
            #self.save(save_dir, save_prefix + '_' + str(epoch))
            '''
            if evaluate:
                self.logger.info('Evaluating the model after epoch {}'.format(epoch))
                if data.dev_set is not None:
                    eval_batches = data.gen_mini_batches('dev', batch_size, pad_id, shuffle=False)
                    bleu_rouge = self.evaluate(eval_batches)
                    #eval_loss, bleu_rouge = self.evaluate(eval_batches)
                    #self.logger.info('Dev eval loss {}'.format(eval_loss))
                    self.logger.info('Dev eval result: {}'.format(bleu_rouge))
                    
                    if bleu_rouge['Bleu-4'] > max_bleu_4:
                        self.save(save_dir, save_prefix)
                        max_bleu_4 = bleu_rouge['Bleu-4']
                    _
                else:
                    self.logger.warning('No dev set is loaded for evaluation in the dataset!')
            else:
                self.save(save_dir, save_prefix + '_' + str(epoch))
            '''



    def evaluate(self, eval_batches, result_dir=None, result_prefix=None, save_full_info=False):
        """
        Evaluates the model performance on eval_batches and results are saved if specified
        Args:
            eval_batches: iterable batch data
            result_dir: directory to save predicted answers, answers will not be saved if None
            result_prefix: prefix of the file for saving predicted answers,
                           answers will not be saved if None
            save_full_info: if True, the pred_answers will be added to raw sample and saved
        """
        pred_answers, ref_answers = [], []
        total_loss, total_num = 0, 0
        for b_itx, batch in enumerate(eval_batches):
            pred_answers, ref_answers = [], []
            print '------ evaluate Batch Question: ' + str(b_itx)
            for sample in batch['raw_data']:
                if 'answers' in sample:
                    ref_answers.append({'question_id': sample['question_id'],
                                         'question_type': sample['question_type'],
                                         'answers': sample['answers'],
                                         'entity_answers': [[]],
                                         'yesno_answers': []})

            print 'ref_answers: '
            print ref_answers
            print batch['padded_p_len']
            p_words_id = []  # all words id
            p_words_list = []  # all words
            l_passages = 1  # include end_pad
            n = 0
            for l in batch['passage_length']:
                l_passages += l
                temp_id = [i + n * (int(batch['padded_p_len'])) for i in range(l)]
                # print temp
                temp_w = batch['passage_token_ids'][n][:l]
                n += 1
                p_words_id += temp_id
                p_words_list += temp_w
            p_words_list.append(0)
            # print p_words_id[-1]
            # print p_words_id
            self.end_pad = []
            self.end_pad.append(p_words_id[-1] + 1)
            p_words_id.append(p_words_id[-1] + 1)
            print 'end_pad: '
            print self.end_pad
            # print p_words_list
            # print p_words_id
            listSelectedSet_id = []
            self.max_a_len = min(self.max_a_len, l_passages)
            feed_dict = {self.p: batch['passage_token_ids'],
                              self.q: [batch['question_token_ids'][0]],
                              self.p_length: batch['passage_length'],
                              self.q_length: [batch['question_length'][0]],
                              self.p_words_id: p_words_id,
                         self.dropout_keep_prob: 1.0}
            # policy
            for tt in xrange(3):
            #for tt in xrange(self.max_a_len):
                max_id = float('-inf')
                policy_c_id = []
                print listSelectedSet_id
                #listSelectedSet_id  = map(eval, listSelectedSet_id)
                for can in listSelectedSet_id:
                    max_id = max(can, max_id)
                for idx in range(l_passages):
                    if idx > max_id:
                        policy_c_id.append(idx)
                if len(listSelectedSet_id) == 0:
                    pred_id = self.sess.run(self.prob_id_first, feed_dict=feed_dict)
                else:
                    feed_dict = dict({self.selected_id_list: listSelectedSet_id,
                                      self.candidate_id: policy_c_id}.items() + feed_dict.items())
                    pred_id = self.sess.run(self.prob_id, feed_dict=feed_dict)
                listSelectedSet_id.append(pred_id)
                #print 'pred_id:'
                #print pred_id
                if pred_id in self.end_pad:
                    print 'break!!!!!!!!!!!'
                    break

            '''
            # value function
            listSelectedSet_id_value = []
            listSelectedSet_id_value = map(eval, listSelectedSet_id_value)
            max_one_value_pred_test = float("-inf")
            one_doc_pred_test = ''
            for ttt in xrange(1):
            # for tt in xrange(self.max_a_len):
            # print words_list
                candidate_list = []
                for can in listSelectedSet_id:
                    max_id = max(can, max_id)
                for idx in range(l_passages):
                    if idx > max_id:
                        candidate_list.append(idx)
                for w_id in candidate_list:
                    c_tmp = listSelectedSet_id_value
                    c_tmp = c_tmp.append(w_id)
                    if len(listSelectedSet_id_value) == 0:
                        value_p = self.sess.run(self.value_first, feed_dict=self.feed_dict)
                    else:
                        feed_dict = dict({self.selected_id_list: listSelectedSet_id_value}.items() + self.feed_dict.items())
                        value_p = self.sess.run(self.value, feed_dict=feed_dict)
                    one_doc_value_pred_test = value_p
                    if one_doc_value_pred_test > max_one_value_pred_test:
                        one_doc_pred_test = w_id
                        max_one_value_pred_test = one_doc_value_pred_test

                listSelectedSet_id_value.append(one_doc_pred_test)
                if one_doc_pred_test in self.end_pad:
                    break
            print 'break!!!!!!!!!!!'
            
            #listSelectedSet_id or listSelectedSet_id_value
            listSelectedSet_words = []
            listSelectedSet = map(eval, listSelectedSet_id)
            for idx in listSelectedSet:
                listSelectedSet_words.append(p_words_list[idx])
            print 'listSelectedSet:'
            print listSelectedSet
            print 'listSelectedSet_words: '
            print listSelectedSet_words
            '''

            for sample in batch['raw_data']:
                #print 'str：'
                strr123 = self.vocab.recover_from_ids(listSelectedSet_id, 0)
                #print strr123
                pred_answers.append({'question_id': sample['question_id'],
                                     'question_type': sample['question_type'],
                                     'answers': [strr123],
                                     'entity_answers': [[]],
                                     'yesno_answers': []})

            print 'pred_answer: '
            print pred_answers

            if len(ref_answers) > 0:
                #print ref_answers
                #print ref_answers
                pred_dict, ref_dict = {}, {}
                for pred, ref in zip(pred_answers, ref_answers):
                    question_id = ref['question_id']
                    if len(ref['answers']) > 0:
                        pred_dict[question_id] = normalize(pred['answers'])
                        ref_dict[question_id] = normalize(ref['answers'])
                bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
            else:
                bleu_rouge = None
            value_with_mcts = bleu_rouge
            
            print 'bleu_rouge(value_with_mcts): '
            print value_with_mcts

        return value_with_mcts

    def find_best_answer(self, sample, start_prob, end_prob, padded_p_len):
        """
        Finds the best answer for a sample given start_prob and end_prob for each position.
        This will call find_best_answer_for_passage because there are multiple passages in a sample
        """
        best_p_idx, best_span, best_score = None, None, 0
        for p_idx, passage in enumerate(sample['passages']):
            if p_idx >= self.max_p_num:
                continue
            passage_len = min(self.max_p_len, len(passage['passage_tokens']))
            answer_span, score = self.find_best_answer_for_passage(
                start_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                end_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                passage_len)
            if score > best_score:
                best_score = score
                best_p_idx = p_idx
                best_span = answer_span
        best_answer = ''.join(
            sample['passages'][best_p_idx]['passage_tokens'][best_span[0]: best_span[1] + 1])
        return best_answer

    def find_best_answer_for_passage(self, start_probs, end_probs, passage_len=None):
        """
        Finds the best answer with the maximum start_prob * end_prob from a single passage
        """
        if passage_len is None:
            passage_len = len(start_probs)
        else:
            passage_len = min(len(start_probs), passage_len)
        best_start, best_end, max_prob = -1, -1, 0
        for start_idx in range(passage_len):
            for ans_len in range(self.max_a_len):
                end_idx = start_idx + ans_len
                if end_idx >= passage_len:
                    continue
                prob = start_probs[start_idx] * end_probs[end_idx]
                if prob > max_prob:
                    best_start = start_idx
                    best_end = end_idx
                    max_prob = prob
        return (best_start, best_end), max_prob

    def save(self, model_dir, model_prefix):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model saved in {}, with prefix {}.'.format(model_dir, model_prefix))

    def restore(self, model_dir, model_prefix):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        self.saver.restore(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model restored from {}, with prefix {}'.format(model_dir, model_prefix))

