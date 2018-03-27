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


class TFGraph(object):
    """
    Implements the main reading comprehension model.

    python -u run.py --train --algo MCST --epochs 10 --batch_size 1 --max_p_len 10000 --hidden_size 150  --train_files ../data/demo/trainset/test10 --dev_files  ../data/demo/devset/test20 --test_files ../data/demo/test/search.test.json
    """

    def __init__(self, name, vocab, args):
        self.tf_name = name
        self.logger = logging.getLogger("brc")
        self.vocab = vocab

        # basic config
        self.algo = args.algo
        self.hidden_size = args.hidden_size
        self.optim_type = args.optim
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.use_dropout = args.dropout_keep_prob < 1

        self._build_graph()

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
        # param_num = sum([np.prod(self.sess.run(tf.shape(v))) for v in self.all_params])
        # self.logger.info('There are {} parameters in the model'.format(param_num))
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

        # test
        self.p_words_id = tf.placeholder(tf.int32, [None])
        self.candidate_id = tf.placeholder(tf.int32, [None])
        # self.words = tf.placeholder(tf.float32, [None, None])
        self.selected_id_list = tf.placeholder(tf.int32, [None])
        self.policy = tf.placeholder(tf.float32, [1, None])  # policy
        self.v = tf.placeholder(tf.float32, [1, 1])  # value

    def _embed(self):
        """
        The embedding layer, question and passage share embeddings
        """
        with tf.device('/cpu:0'), tf.variable_scope('word_embedding'):
            # with tf.variable_scope('word_embedding'):
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
            _, self.sep_q_encodes = rnn('bi-lstm', self.q_emb, self.q_length, self.hidden_size)
        if self.use_dropout:
            self.p_encodes = tf.nn.dropout(self.p_encodes, self.dropout_keep_prob)
            self.sep_q_encodes = tf.nn.dropout(self.sep_q_encodes, self.dropout_keep_prob)

    def _initstate(self):
        self.V = tf.Variable(tf.random_uniform([self.hidden_size * 2, self.hidden_size * 2], -1. / self.hidden_size,
                                               1. / self.hidden_size))
        self.W = tf.Variable(
            tf.random_uniform([self.hidden_size * 2, 1], -1. / self.hidden_size, 1. / self.hidden_size))

        self.W_b = tf.Variable(tf.random_uniform([1, 1], -1. / self.hidden_size, 1. / self.hidden_size))

        self.V_c = tf.Variable(
            tf.random_uniform([self.hidden_size * 2, self.hidden_size], -1. / self.hidden_size, 1. / self.hidden_size))
        self.V_h = tf.Variable(
            tf.random_uniform([self.hidden_size * 2, self.hidden_size], -1. / self.hidden_size, 1. / self.hidden_size))

        self.q_state_c = tf.sigmoid(tf.matmul(self.sep_q_encodes, self.V_c))
        self.q_state_h = tf.sigmoid(tf.matmul(self.sep_q_encodes, self.V_h))
        self.q_state = tf.concat([self.q_state_c, self.q_state_h], 1)

        self.shape_a = tf.shape(self.q_state)
        self.shape_b = tf.shape(self.p_encodes)

        self.words = tf.reshape(self.p_encodes, [-1, self.hidden_size * 2])

        # self.words_list = tf.gather(self.words, self.p_words_id) # all words in a question doc

    def _action_frist(self):
        """
        select first word
        """
        # self.candidate = tf.reshape(self.p_emb,[-1,self.hidden_size*2])
        self.logits_first = tf.reshape(tf.matmul(tf.matmul(self.words, self.V), tf.transpose(self.q_state)), [-1])
        self.prob_first = tf.nn.softmax(self.logits_first)
        self.prob_id_first = tf.argmax(self.prob_first)
        self.value_first = tf.sigmoid(tf.reshape(tf.matmul(self.q_state, self.W), [1, 1]) + self.W_b)  # [1,1]

    def _action(self):
        """
        Employs Bi-LSTM again to fuse the context information after match layer
        """
        self.candidate = tf.gather(self.words, self.candidate_id)
        self.selected_list = tf.gather(self.words, self.selected_id_list)
        self.input = tf.reshape(self.selected_list, [1, -1, self.hidden_size * 2])
        rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_size, state_is_tuple=False)
        _, self.states = tf.nn.dynamic_rnn(rnn_cell, self.input, initial_state=self.q_state,
                                           dtype=tf.float32)  # [1, dim]
        self.logits = tf.reshape(tf.matmul(tf.matmul(self.candidate, self.V), tf.transpose(self.states)), [-1])

        self.prob = tf.nn.softmax(self.logits)
        self.prob_id = tf.argmax(self.prob)
        self.value = tf.sigmoid(tf.reshape(tf.matmul(self.states, self.W), [1, 1]) + self.W_b)  # [1,1]

    def value_function(self, words_list):
        words_list = map(eval, words_list)
        # print words_list
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
            max_id = max(can, max_id)
        for idx in range(l_passages):
            if idx > max_id:
                policy_c_id.append(idx)
        if len(words_list) == 0:
            c_pred = self.sess.run(self.prob_first, feed_dict=self.feed_dict)
        else:
            feed_dict = dict(
                {self.selected_id_list: words_list, self.candidate_id: policy_c_id}.items() + self.feed_dict.items())
            c_pred = self.sess.run(self.prob, feed_dict=feed_dict)

        return policy_c_id, c_pred


    def _compute_loss(self):
        """
        The loss function
        """
        self.loss_first = tf.contrib.losses.mean_squared_error(self.v, self.value_first) - tf.matmul(self.policy,tf.reshape(tf.log(
                                                             tf.clip_by_value(self.prob_first, 1e-30,1.0)),[-1, 1]))
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
    def set_feed_dict(self,p,q,p_length,q_length,dropout_keep_prob):
        self.feed_dict = {self.p: p,
                          self.q: q,
                          self.p_length: p_length,
                          self.q_length: q_length,
                          self.dropout_keep_prob: dropout_keep_prob}
    def cal_first_loss(self, policy, input_v):
        feed_dict = dict(self.feed_dict.items() + {self.policy: [policy], self.v: [[input_v]]}.items())
        _, loss = self.sess.run([self.optimizer_first, self.loss_first], feed_dict=feed_dict)
        return loss
    def cal_loss(self, policy, input_v,selected_id_list,c):
        feed_dict = dict(self.feed_dict.items() + {self.selected_id_list: selected_id_list, self.candidate_id: c,
                                      self.policy: [policy],self.v: [[input_v]]}.items())
        _, loss = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
        return loss
    def run_session_shape(self):
        shape_a, shape_b = self.sess.run([self.shape_a, self.shape_b], feed_dict=self.feed_dict)
        return shape_a,shape_b
