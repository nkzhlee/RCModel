# !/usr/bin/python
# -*- coding:utf-8 -*-

import multiprocessing as mp
import tensorflow as tf
from sub_tree import sub_tree
from sub_tree import node
import sys
import logging
import time
import Queue
import numpy as np

from treelib import Tree
import copy
from utils import compute_bleu_rouge
from utils import normalize
from layers.basic_rnn import rnn
from layers.match_layer import MatchLSTMLayer
from layers.match_layer import AttentionFlowMatchLayer
from layers.pointer_net import PointerNetDecoder


def main():
    for idx in range(10):
        print idx


def job(x):
    return x * x


'''
def tree_search(trees_batch):
    pool = mp.Pool()
    print("Number of cpu : ", mp.cpu_count())
    res = pool.map(job, range(10))
    print res


def tree_search(trees_batch):
    mp.log_to_stderr()
    logger = mp.get_logger()
    logger.setLevel(logging.INFO)
    print("Number of cpu : ", mp.cpu_count())
    procs = []
    queue = mp.Queue()
    #print 'procs  '+ str(len(procs))
    for t in trees_batch:
        proc = mp.Process(target = sub_tree, args = (t,))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()
'''


def test_tf():
    x = tf.placeholder(tf.float64, shape=None)
    y = tf.placeholder(tf.float64, shape=None)
    z = tf.placeholder(tf.float64, shape=None)
    a = np.ones((1, 5, 4))
    b = np.array([[[1, 2], [1, 3]], [[0, 1], [0, 2]]])
    c = np.array([[(1., 2., 3.), (2., 3., 4.), (3., 4., 5.), (4., 5., 6.)],
                  [(1., 2., 3.), (2., 2., 2.), (3., 4., 5.), (4., 5., 6.)]])
    # print a
    print b
    print c
    print type(b)
    # y = tf.multiply(x,)
    tmp = tf.expand_dims(z, 0)
    sa = tf.shape(x)
    sb = tf.shape(y)
    sc = tf.shape(z)
    s = tf.shape(tmp)

    # q = tf.matmul(x, tmp)
    # sd = tf.shape(q)

    r = tf.gather_nd(c, b)
    sr = tf.shape(r)
    # print np.shape(a)
    # print np.shape(b)
    with tf.Session() as sess:
        sb, sc, s, tmp, r, sr = sess.run([sb, sc, s, tmp, r, sr], feed_dict={x: a, y: b, z: c})
    print sb
    print sc
    # print q
    print r
    print sr
    # return result


class Data_tree(object):
    def __init__(self, tree, start_node):
        self.tree = tree
        self.start_node = start_node
        self.q_id = tree.raw_tree_data['tree_id']
        self.q_type = tree.raw_tree_data['question_type']
        self.words_id_list = tree.raw_tree_data['passage_token_id']
        self.l_passage = tree.raw_tree_data['p_length']
        self.ref_answer = tree.raw_tree_data['ref_answer']
        self.p_data = []
        self.listSelectedSet = []

        self.value = 0
        self.select_list = []

        self.p_word_id, self.p_pred = [], []

        self.tmp_node = None
        self.expand_node = None
        self.num_of_search = 0


class PSCHTree(object):
    """
    python -u run.py --train --algo MCST --epochs 1 --gpu 2 --max_p_len 2000 --hidden_size 150  --train_files ../data/demo/trainset/search.train.json --dev_files  ../data/demo/devset/search.dev.json --test_files ../data/demo/test/search.test.json
    nohup python -u run.py --train --algo BIDAF --epochs 10  --train_files ../data/demo/trainset/test_5 --dev_files  ../data/demo/devset/test_5 --test_files ../data/demo/test/search.test.json >test5.txt 2>&1 &
    nohup python -u run.py --train --algo MCST --epochs 100 --gpu 3 --max_p_len 1000 --hidden_size 150  --train_files ../data/demo/trainset/search.train.json --dev_files  ../data/demo/devset/search.dev.json --test_files ../data/demo/test/search.test.json >test_313.txt 2>&1 &


    """

    def __init__(self, args, vocab):

        self.vocab = vocab
        # logging

        self.logger = logging.getLogger("brc")

        # basic config
        self.algo = args.algo
        self.hidden_size = args.hidden_size
        self.optim_type = args.optim
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.use_dropout = args.dropout_keep_prob < 1
        self.dropout_keep_prob = 1.0

        # length limit
        self.max_p_num = args.max_p_num
        self.max_p_len = args.max_p_len
        self.max_q_len = args.max_q_len
        # self.max_a_len = args.max_a_len
        self.max_a_len = 2
        # test paras
        self.search_time = 3
        self.beta = 100.0

        self._build_graph()

    def _init_sub_tree(self, tree):
        print '------- init sub tree :' + str(tree['tree_id']) + '---------'
        start_node = 'question_' + str(tree['tree_id'])
        mcts_tree = sub_tree(tree)
        data_tree = Data_tree(mcts_tree, start_node)
        data_tree.num_of_search += 1
        return data_tree

    def _do_init_tree_job(self, lock, trees_to_accomplish, trees_that_are_done, log):
        while True:
            try:
                '''
                    try to get task from the queue. get_nowait() function will 
                    raise queue.Empty exception if the queue is empty. 
                    queue(False) function would do the same task also.
                '''
                with lock:
                    tree = trees_to_accomplish.get_nowait()
            except Queue.Empty:

                break
            else:
                '''
                    if no exception has been raised, add the task completion 
                    message to task_that_are_done queue
                '''
                # result = self._init_sub_tree(tree)
                print '------- init sub tree :' + str(tree['tree_id']) + '---------'
                start_node = 'question_' + str(tree['tree_id'])
                mcts_tree = sub_tree(tree)
                data_tree = Data_tree(mcts_tree, start_node)
                data_tree.num_of_search += 1
                lock.acquire()
                try:
                    log.put(str(tree['tree_id']) + ' is done by ' + str(mp.current_process().name))
                    trees_that_are_done.put(data_tree)
                finally:
                    lock.release()
                    # time.sleep(.5)
        return True

    def _search_sub_tree(self, data_tree):
        sub_tree = data_tree.tree
        # print '------- search sub tree :' + str(sub_tree.q_id) + '---------'
        start_node_id = data_tree.start_node
        data_tree.num_of_search += 1
        data_tree.select_list = [start_node_id]
        tmp_node = sub_tree.tree.get_node(start_node_id)
        while not tmp_node.is_leaf():
            max_score = float("-inf")
            max_id = -1
            for child_id in tmp_node.fpointer:
                child_node = sub_tree.tree.get_node(child_id)
                # score = child_node.data.p
                score = self.beta * child_node.data.p * ((1 + sub_tree.count) / (1 + child_node.data.num))
                if score > max_score:
                    max_id = child_id
                    max_score = score
            data_tree.select_list.append(max_id)
            tmp_node = sub_tree.tree.get_node(max_id)
            data_tree.tmp_node = tmp_node
        return data_tree

    def _do_search_tree_job(self, lock, trees_to_accomplish, trees_that_are_done, log):
        while True:
            try:
                '''
                    try to get task from the queue. get_nowait() function will 
                    raise queue.Empty exception if the queue is empty. 
                    queue(False) function would do the same task also.
                '''
                with lock:
                    data_tree = trees_to_accomplish.get_nowait()
            except Queue.Empty:
                break
            else:
                '''
                    if no exception has been raised, add the task completion 
                    message to task_that_are_done queue
                '''
                # result = self._search_sub_tree(tree)
                sub_tree = data_tree.tree
                # print '------- search sub tree :' + str(sub_tree.q_id) + '---------'
                start_node_id = data_tree.start_node
                data_tree.num_of_search += 1
                data_tree.select_list = [start_node_id]
                tmp_node = sub_tree.tree.get_node(start_node_id)
                while not tmp_node.is_leaf():
                    max_score = float("-inf")
                    max_id = -1
                    for child_id in tmp_node.fpointer:
                        child_node = sub_tree.tree.get_node(child_id)
                        # score = child_node.data.p
                        score = self.beta * child_node.data.p * ((1 + sub_tree.count) / (1 + child_node.data.num))
                        if score > max_score:
                            max_id = child_id
                            max_score = score
                    data_tree.select_list.append(max_id)
                    tmp_node = sub_tree.tree.get_node(max_id)
                    data_tree.tmp_node = tmp_node
                lock.acquire()
                try:
                    log.put(str(data_tree.tmp_node) + ' is selected by ' + str(mp.current_process().name))
                    # print str(data_tree.tmp_node) + ' is selected by ' + str(mp.current_process().name)
                    trees_that_are_done.put(data_tree)
                finally:
                    lock.release()

        return True

    def _aciton_tree(self, data_tree):
        start_node = data_tree.start_node
        tmp_policy = self._get_policy(data_tree)
        # print (tmp_policy.values())
        # print (sum(tmp_policy.values()))
        prob, select_word_id, start_node = self._take_action(data_tree)
        data_tree.p_data.append(prob)
        data_tree.listSelectedSet.append(select_word_id)
        return data_tree

    def _do_tree_action_job(self, lock, trees_to_accomplish, action_result_queue, log):
        while True:
            try:
                '''
                    try to get task from the queue. get_nowait() function will 
                    raise queue.Empty exception if the queue is empty. 
                    queue(False) function would do the same task also.
                '''
                lock.acquire()
                try:
                    data_tree = trees_to_accomplish.get_nowait()
                finally:
                    lock.release()
            except Queue.Empty:

                break
            else:
                '''
                    if no exception has been raised, add the task completion 
                    message to task_that_are_done queue
                '''
                # result = self._aciton_tree(tree)
                # result = tree
                prob, select_word_id, start_node = self._take_action(data_tree)
                data_tree.start_node = start_node
                data_tree.p_data.append(prob)
                # print ('data_tree.listSelectedSet',data_tree.listSelectedSet)
                data_tree.listSelectedSet.append(select_word_id)
                lock.acquire()
                try:
                    log.put(str(data_tree.listSelectedSet) + ' is list of action choosen by ' + str(
                        mp.current_process().name))
                    action_result_queue.put(data_tree)
                finally:
                    lock.release()
        return True

    def feed_in_batch(self, tree_batch, parallel_size, feed_dict):
        self.tree_batch = tree_batch
        self.para_size = parallel_size
        self.batch_size = len(self.tree_batch['tree_ids'])
        # self.feed_dict = feed_dict

    def tree_search(self):
        trees = []
        test_tf()

        # 1)initialize trees
        for bitx in range(self.batch_size):
            # print '-------------- yeild ' + str(bitx) + '-------------'
            if self.tree_batch['p_length'][bitx] > self.max_p_len:
                # print '>>>>>>>>>>>>>>>> '
                self.tree_batch['p_length'][bitx] = self.max_p_len
                self.tree_batch['candidates'][bitx] = self.tree_batch['candidates'][bitx][:(self.max_p_len)]  # ???
            tree = {'tree_id': self.tree_batch['tree_ids'][bitx],
                    'question_token_ids': self.tree_batch['root_tokens'][bitx],
                    'passage_token_id': self.tree_batch['candidates'][bitx],
                    'q_length': self.tree_batch['q_length'][bitx],
                    'p_length': self.tree_batch['p_length'][bitx],
                    'question_type': self.tree_batch['question_type'][bitx],
                    'ref_answer': self.tree_batch['ref_answers'][bitx]
                    # 'mcst_model':self.tree_batch['mcst_model']
                    }
            trees.append(tree)

        print ('Max parallel processes size: ', self.para_size)
        number_of_task = self.batch_size
        number_of_procs = self.para_size
        manager = mp.Manager()
        trees_to_accomplish = manager.Queue()
        trees_that_are_done = manager.Queue()
        log = mp.Queue()
        processes = []
        lock = manager.Lock()
        for i in trees:
            trees_to_accomplish.put(i)

        # creating processes
        for w in range(number_of_procs):
            p = mp.Process(target=self._do_init_tree_job, args=(lock, trees_to_accomplish, trees_that_are_done, log))
            processes.append(p)
            p.start()

        # completing process
        for p in processes:
            p.join()
        while not log.empty():
            print(log.get())

        # for i,p in enumerate(processes):
        #     if not p.is_alive():
        #         print ("[MAIN]: WORKER is a goner", i)

        # init the root node and expand the root node

        self.tree_list = []
        init_list = []
        while not trees_that_are_done.empty():
            now_tree = trees_that_are_done.get()
            now_tree.expand_node = now_tree.tree.tree.get_node(now_tree.tree.tree.root)
            init_list.append(now_tree)
            # self._init_root(now_tree)
            # self.tree_list.append(now_tree)
        # init_roots(init_list)
        self.tree_list = self.expands(init_list)
        # search tree
        self.end_tree = []
        for t in xrange(self.max_a_len):
            print ('Answer_len', t)
            if len(self.tree_list) == 0:
                break
            for s_time in range(self.search_time):
                print ('search time', s_time)

                # creating processes
                processes_search = []

                tree_search_queue = manager.Queue()
                tree_result_queue = manager.Queue()

                for tree in self.tree_list:
                    tree_search_queue.put(tree)

                search_tree_list = []

                for w in range(number_of_procs):
                    p = mp.Process(target=self._do_search_tree_job,
                                   args=(lock, tree_search_queue, tree_result_queue, log))
                    processes_search.append(p)
                    p.start()
                    time.sleep(0.1)

                while 1:
                    if not tree_result_queue.empty():
                        data_tree = tree_result_queue.get()
                        search_tree_list.append(data_tree)
                    if len(search_tree_list) == number_of_procs:
                        break
                        # time.sleep(0.1)

                # completing process
                for p in processes_search:
                    # p.join()
                    p.terminate()

                while not log.empty():
                    print(log.get())

                self.tree_list = []
                # gather train data

                self.tree_list = self._search_vv(search_tree_list)

                tree_need_expand_list = []
                tree_no_need_expand_list = []
                for data_tree in self.tree_list:
                    data_tree_update = self._updates(data_tree)
                    tmp_node = data_tree_update.tmp_node
                    l_passage = data_tree_update.l_passage
                    word_id = int(tmp_node.data.word[-1])
                    if tmp_node.is_leaf() and (word_id < (l_passage - 1)):
                        data_tree_update.expand_node = tmp_node
                        tree_need_expand_list.append(data_tree_update)
                    else:
                        tree_no_need_expand_list.append(data_tree_update)

                self.tree_list = self.expands(tree_need_expand_list)
                self.tree_list.append(tree_no_need_expand_list)

                # for data_tree in search_tree_list:
                #     value = self._search_v(data_tree)
                #     data_tree_update = self._update(data_tree, value)
                #     data_tree_update.tree.count += 1
                #     tmp_node = data_tree_update.tmp_node
                #     l_passage = data_tree_update.l_passage
                #     word_id = int(tmp_node.data.word[-1])
                #     if tmp_node.is_leaf() and (word_id < (l_passage-1)):
                #         tree_data = data_tree_update.tree.get_raw_tree_data()
                #         feed_dict = {self.p: [tree_data['passage_token_id']],
                #                      self.q: [tree_data['question_token_ids']],
                #                      self.p_length: [tree_data['p_length']],
                #                      self.q_length: [tree_data['q_length']],
                #                      self.dropout_keep_prob: 1.0}
                #         data_tree_update = self.expand(data_tree_update,tmp_node,feed_dict)
                #     self.tree_list.append(data_tree_update)
            # take action


            num_action_procs = 0

            action_queue = manager.Queue()
            action_result_queue = manager.Queue()
            for tree in self.tree_list:
                # print ('######### tree.listSelectedSet: ', tree.listSelectedSet)
                # print ('num ', tree.num_of_search)
                if not len(tree.listSelectedSet) == 0:
                    last_word = tree.listSelectedSet[-1]
                    if not last_word == (tree.l_passage - 1):
                        action_queue.put(tree)
                        num_action_procs += 1
                    else:
                        self.end_tree.append(tree)
                else:
                    action_queue.put(tree)
                    num_action_procs += 1
            action_tree_list = []
            processes_action = []
            # print ('###start take action ')
            # print ('len(self.tree_list)', len(self.tree_list))
            '''
            for w in range(num_action_procs):
                #print (w, w)
                p = mp.Process(target=self._do_tree_action_job, args=(lock, action_queue, action_result_queue, log))
                processes_action.append(p)
                p.start()
                #time.sleep(0.1)
            # completing process
            while 1:
                #time.sleep(0.1)
                if not action_result_queue.empty():
                    data_tree = action_result_queue.get()
                    action_tree_list.append(data_tree)
                if len(action_tree_list) == num_action_procs:
                    break

            for p in processes_action:
                p.terminate()

            while not log.empty():
                print(log.get())

            self.tree_list = action_tree_list
            for selection in action_tree_list:
                print ('selection', selection.listSelectedSet)

        for t in self.tree_list:
            self.end_tree.append(t)

        print ('----end tree:', len(self.end_tree))
        #create nodes --->search  until finish ----
        pred_answers,ref_answers = [],[]
        for tree in self.tree_list:
            p_words_list = tree.words_id_list
            listSelectedSet_words = []
            listSelectedSet = map(eval, tree.listSelectedSet)
            for idx in listSelectedSet:
                listSelectedSet_words.append(p_words_list[idx])
                # print 'listSelectedSet:'
                #print listSelectedSet
                # print 'listSelectedSet_words: '
                # print listSelectedSet_words'
            strr123 = self.vocab.recover_from_ids(listSelectedSet_words, 0)
            # print strr123
            pred_answers.append({'question_id': tree.q_id,
                                 'question_type': tree.q_type,
                                 'answers': [''.join(strr123)],
                                 'entity_answers': [[]],
                                 'yesno_answers': []})
            ref_answers.append({'question_id': tree.q_id,
                             'question_type': tree.q_type,
                             'answers': tree.ref_answer,
                             'entity_answers': [[]],
                             'yesno_answers': []})

        #print 'pred_answer: '
        #print pred_answers

        #print 'ref_answers: '
        #print ref_answers

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

        for tree in self.end_tree:
            tree_data = tree.tree.get_raw_tree_data()
            input_v = value_with_mcts['Bleu-4']
            total_num, total_loss = 0, 0
            log_every_n_batch, n_batch_loss = 3, 0
            for prob_id, prob_data in enumerate(tree.p_data):
                # print 'p_data: '
                # print prob_id
                # print prob_data
                c = []
                policy = []
                for prob_key, prob_value in prob_data.items():
                    c.append(prob_key)
                    policy.append(prob_value)
                print 'policy: '
                print [policy]
                print 'value: '
                print [value_with_mcts]
                print 'candidate: '
                print c
                print 'listSelectedSet[:prob_id]'
                print listSelectedSet[:prob_id]
                input_v = value_with_mcts['Rouge-L']

                feed_dict = {self.p: [tree_data['passage_token_id']],
                             self.q: [tree_data['question_token_ids']],
                             self.p_length: [tree_data['p_length']],
                             self.q_length: [tree_data['q_length']],
                             self.dropout_keep_prob: 1.0}
                if prob_id == 0:
                    feeddict = dict(feed_dict.items() + {self.policy: [policy], self.v: [[input_v]]}.items())
                    loss_first = self.sess.run([self.loss_first], feed_dict=feeddict)
                    print('loss,first', loss_first)
                else:
                    feeddict = dict(feed_dict.items() + {self.selected_id_list: [listSelectedSet[:prob_id]], self.candidate_id: [c],
                                                  self.policy: [policy],
                                                  self.v: [[input_v]]}.items())
                    loss = self.sess.run([self.loss], feed_dict=feeddict)
                    print('loss',loss)

            total_loss += loss * len(self.end_tree)
            total_num += len(self.end_tree)
            n_batch_loss += loss
            if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                self.logger.info('Average loss from batch {} to {} is {}'.format(
                    bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch))
            n_batch_loss = 0

        return 1.0 * total_loss / total_num

        '''

        return 0

    def _init_root(self, now_tree):
        tree = now_tree.tree
        tree_data = tree.get_raw_tree_data()
        # print ('start_node ', start_node)
        feed_dict = {self.p: [tree_data['passage_token_id']],
                     self.q: [tree_data['question_token_ids']],
                     self.p_length: [tree_data['p_length']],
                     self.q_length: [tree_data['q_length']],
                     self.dropout_keep_prob: 1.0}
        leaf_node = tree.tree.get_node(tree.tree.root)
        self.expand(now_tree, leaf_node, feed_dict)

    def expands(self, tree_list):
        print ('========== start expands ==============')
        p_feed = []
        q_feed = []
        p_lenth_feed = []
        q_length_feed = []
        words_list_list = []
        l_passage_list = []
        policy_need_list = []
        for t_idx, data_tree in enumerate(tree_list, 0):
            tree_data = data_tree.tree.get_raw_tree_data()
            word_list = data_tree.expand_node.data.word
            l_passage = data_tree.l_passage
            print ('1word_list', word_list)
            if (len(word_list) == 0):
                data_tree = self._get_init_policy(data_tree, l_passage)
            else:
                p_feed.append(tree_data['passage_token_id'])
                q_feed.append(tree_data['question_token_ids'])
                p_lenth_feed.append(tree_data['p_length'])
                q_length_feed.append(tree_data['q_length'])
                words_list_list.append(data_tree.expand_node.data.word)
                l_passage_list.append(data_tree.l_passage)
                policy_need_list.append(t_idx)

        if not (len(p_feed) == 0):
            feed_dict = {self.p: p_feed,
                         self.q: q_feed,
                         self.p_length: p_lenth_feed,
                         self.q_length: q_length_feed,
                         self.dropout_keep_prob: 1.0}
            policy_ids, policys = self._cal_policys(words_list_list, l_passage_list, feed_dict)
            for p_idx, t_idx in enumerate(policy_need_list, 0):
                tree_list[t_idx].p_pred = policys[p_idx]
                tree_list[t_idx].p_word_id = policys[p_idx]
        for d_tree in tree_list:
            print ('d_tree.p_pred ', np.shape(d_tree.p_pred))
            print ('d_tree.p_word_id', np.shape(d_tree.p_word_id))
            leaf_node = d_tree.expand_node
            words_list = leaf_node.data.word
            print ('words_list', words_list)
            for word in d_tree.p_word_id:
                d_tree.tree.node_map[' '.join(words_list + [str(word)])] = len(d_tree.tree.node_map)
                new_node = node()
                new_node.word = words_list + [str(word)]
                idx = d_tree.p_word_id.index(word)
                new_node.p = d_tree.p_pred[idx]
                # print 'new_node.p ' + str(new_node.p)
                d_tree.tree.tree.create_node(identifier=d_tree.tree.node_map[' '.join(new_node.word)], data=new_node,
                                             parent=leaf_node.identifier)
        print ('========== end expands ==============')
        return tree_list

    def _search_v(self, data_tree):
        tree = data_tree.tree
        tree_data = tree.get_raw_tree_data()
        feed_dict = {self.p: [tree_data['passage_token_id']],
                     self.q: [tree_data['question_token_ids']],
                     self.p_length: [tree_data['p_length']],
                     self.q_length: [tree_data['q_length']],
                     self.dropout_keep_prob: 1.0}
        l_passage = tree_data['p_length']
        tmp_node = data_tree.tmp_node
        # print('tmp_node', str(tmp_node))
        # print('tmp_node.data.word[-1]: ', int(tmp_node.data.word[-1]))
        # print('l_passage', l_passage)
        word_id = int(tmp_node.data.word[-1])
        words_list = tmp_node.data.word
        if (word_id == (l_passage - 1)):
            v = 0
            pred_answer = tmp_node.data.word
            # print 'pred_answer: '
            # print pred_answer
            # print 'listSelectedSet'
            listSelectedSet_words = []
            listSelectedSet = map(eval, pred_answer)
            # print listSelectedSet
            for idx in listSelectedSet:
                listSelectedSet_words.append(data_tree.words_id_list[idx])
            # print 'str123'
            str123 = self.vocab.recover_from_ids(listSelectedSet_words, 0)
            # print str123
            pred_answers = []
            ref_answers = []
            pred_answers.append({'question_id': data_tree.q_id,
                                 'question_type': data_tree.q_type,
                                 'answers': [''.join(str123)],
                                 'entity_answers': [[]],
                                 'yesno_answers': []})
            ref_answers.append({'question_id': data_tree.q_id,
                                'question_type': data_tree.q_type,
                                'answers': data_tree.ref_answer,
                                'entity_answers': [[]],
                                'yesno_answers': []})
            print '****tree_search'
            # print 'pred_answer: '
            # print pred_answers

            # print 'ref_answers: '
            # print ref_answers
            if len(data_tree.ref_answer) > 0:
                pred_dict, ref_dict = {}, {}
                for pred, ref in zip(pred_answers, ref_answers):
                    question_id = ref['question_id']
                    if len(ref['answers']) > 0:
                        pred_dict[question_id] = normalize(pred['answers'])
                        ref_dict[question_id] = normalize(ref['answers'])


                        # print '========compare in tree======='
                        # print pred_dict[question_id]
                        # print '----------------------'
                        # print ref_dict[question_id]
                bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
            else:
                bleu_rouge = None
            # print 'bleu_rouge'
            # print bleu_rouge
            v = bleu_rouge['Bleu-4']
            print ('v: ', v)
        else:
            v = self._cal_value(words_list, feed_dict)
        return v

    def _get_init_policy(self, data_tree, l_passage):
        print('&&&&&&&&& start init_policy &&&&&&&&')
        tree = data_tree.tree
        tree_data = tree.get_raw_tree_data()
        feed_dict = {self.p: [tree_data['passage_token_id']],
                     self.q: [tree_data['question_token_ids']],
                     self.p_length: [tree_data['p_length']],
                     self.q_length: [tree_data['q_length']],
                     self.dropout_keep_prob: 1.0}
        print ('length of passage', tree_data['p_length'])
        print ('length of padding passage', len(tree_data['passage_token_id']))
        print ('padding', tree_data['passage_token_id'][-1])

        data_tree.p_pred = self.sess.run(self.prob_first, feed_dict=feed_dict)
        data_tree.p_word_id = [i for i in range(l_passage)]
        print('&&&&&&&&& end init_policy &&&&&&&&')
        return data_tree

    def _get_init_value(self, data_tree):

        print('$$$$$$$ start init_value $$$$$$$$$')
        tree = data_tree.tree
        tree_data = tree.get_raw_tree_data()
        feed_dict = {self.p: [tree_data['passage_token_id']],
                     self.q: [tree_data['question_token_ids']],
                     self.p_length: [tree_data['p_length']],
                     self.q_length: [tree_data['q_length']],
                     self.dropout_keep_prob: 1.0}

        value_p = self.sess.run(self.value_first, feed_dict=feed_dict)
        print ('_get_init_value', value_p)
        print('$$$$$$$ end init_value $$$$$$$$$')
        return value_p

    def _search_vv(self, search_tree_list):
        print('-------------------- start search_vv -----------------------')
        value_id_list = []
        p_feed = []
        q_feed = []
        p_lenth_feed = []
        q_length_feed = []
        words_list_list = []
        for t_id, data_tree in enumerate(search_tree_list, 0):
            tree_data = data_tree.tree.get_raw_tree_data()
            tmp_node = data_tree.tmp_node
            word_id = int(tmp_node.data.word[-1])
            l_passage = tree_data['p_length']
            words_list = tmp_node.data.word
            if len(words_list) == 0:
                data_tree.value = self._get_init_value(data_tree)
            else:
                print ('word_id', word_id)
                if (word_id == (l_passage - 1)):
                    v = 0
                    pred_answer = tmp_node.data.word
                    listSelectedSet_words = []
                    listSelectedSet = map(eval, pred_answer)
                    # print listSelectedSet
                    for idx in listSelectedSet:
                        listSelectedSet_words.append(data_tree.words_id_list[idx])
                    str123 = self.vocab.recover_from_ids(listSelectedSet_words, 0)
                    pred_answers = []
                    ref_answers = []
                    pred_answers.append({'question_id': data_tree.q_id,
                                         'question_type': data_tree.q_type,
                                         'answers': [''.join(str123)],
                                         'entity_answers': [[]],
                                         'yesno_answers': []})
                    ref_answers.append({'question_id': data_tree.q_id,
                                        'question_type': data_tree.q_type,
                                        'answers': data_tree.ref_answer,
                                        'entity_answers': [[]],
                                        'yesno_answers': []})
                    print '****tree_search'
                    if len(data_tree.ref_answer) > 0:
                        pred_dict, ref_dict = {}, {}
                        for pred, ref in zip(pred_answers, ref_answers):
                            question_id = ref['question_id']
                            if len(ref['answers']) > 0:
                                pred_dict[question_id] = normalize(pred['answers'])
                                ref_dict[question_id] = normalize(ref['answers'])
                        bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
                    else:
                        bleu_rouge = None
                    v = bleu_rouge['Bleu-4']
                    print ('v: ', v)
                    data_tree.v = v
                else:
                    p_feed.append(np.array(tree_data['passage_token_id']))
                    q_feed.append(np.array(tree_data['question_token_ids']))
                    p_lenth_feed.append(np.array(tree_data['p_length']))
                    q_length_feed.append(np.array(tree_data['q_length']))
                    words_list_list.append(words_list)
                    value_id_list.append(t_id)
        if not (len(p_feed)) == 0:
            self.feed_dict = {self.p: p_feed,
                              self.q: q_feed,
                              self.p_length: p_lenth_feed,
                              self.q_length: q_length_feed,
                              self.dropout_keep_prob: 1.0}
            values = self._cal_values(words_list_list, self.feed_dict)

        for t_idx, v_idx in enumerate(value_id_list, 0):
            search_tree_list[v_idx].value = values[t_idx]

        print('---------------------- end search_vv -----------------------')

        return search_tree_list

    def _cal_values(self, words_list_list, feeddict):
        fd_words_list = []
        seq_length = []
        for idx, words_list in enumerate(words_list_list, 0):
            words_list = map(eval, words_list)
            tp = []
            for word in words_list:
                tp = np.array([idx, word])
            fd_words_list.append(tp)
            seq_length.append(np.array(len(words_list)))

        fd_words_list = np.array(fd_words_list)
        seq_length = np.array(seq_length)

        print ('fd_words_list', fd_words_list)
        print ('shape : ', np.shape(fd_words_list))

        print ('seq_length', seq_length)
        print ('shape : ', np.shape(seq_length))

        print feeddict

        # print ('seq_length', seq_length)

        feed_dict = dict({self.selected_id_list: fd_words_list, self.seq_length: seq_length,
                          self.selected_batch_size: len(seq_length)}.items() + feeddict.items())
        values = self.sess.run(self.value, feed_dict=feed_dict)

        # feed_dict = dict({self.seq_length: seq_length}.items() + feeddict.items())
        # values = self.sess.run(self.shape_a, feed_dict=feed_dict)


        print ('values', values)
        return values

    def _update(self, data_tree, value):
        node_list = data_tree.select_list
        for node_id in node_list:
            tmp_node = data_tree.tree.tree.get_node(node_id)
            tmp_node.data.Q = (tmp_node.data.Q * tmp_node.data.num + value) / (tmp_node.data.num + 1)
            tmp_node.data.num += 1
        return data_tree

    def _updates(self, data_tree):
        node_list = data_tree.select_list
        value = data_tree.value
        for node_id in node_list:
            tmp_node = data_tree.tree.tree.get_node(node_id)
            tmp_node.data.Q = (tmp_node.data.Q * tmp_node.data.num + value) / (tmp_node.data.num + 1)
            tmp_node.data.num += 1
        data_tree.tree.count += 1
        return data_tree

    def _get_policy(self, data_tree):
        sub_tree = data_tree.tree
        start_node_id = data_tree.start_node
        tmp_node = sub_tree.tree.get_node(start_node_id)
        max_time = -1
        prob = {}
        for child_id in tmp_node.fpointer:
            child_node = sub_tree.tree.get_node(child_id)
            if sub_tree.count == 0:
                prob[child_node.data.word[-1]] = 0.0
            else:
                prob[child_node.data.word[-1]] = child_node.data.num / sub_tree.count
        return prob

    def expand(self, data_tree, leaf_node, feed_dict):
        print '======= expand: '
        words_list = leaf_node.data.word
        print 'word_list:'
        print words_list
        l_passage = data_tree.l_passage
        tree = data_tree.tree
        p_word_id, p_pred = self._cal_policy(words_list, l_passage, feed_dict)
        # print 'candidate_id: '
        # print p_word_id
        for word in p_word_id:
            tree.node_map[' '.join(words_list + [str(word)])] = len(tree.node_map)

            new_node = node()
            new_node.word = words_list + [str(word)]

            new_node.p = p_pred[p_word_id.index(word)]
            # print 'new_node.p ' + str(new_node.p)
            tree.tree.create_node(identifier=tree.node_map[' '.join(new_node.word)], data=new_node,
                                  parent=leaf_node.identifier)
        data_tree.tree = tree
        return data_tree

    def _take_action(self, data_tree):
        sub_tree = data_tree.tree
        start_node_id = data_tree.start_node
        tmp_node = sub_tree.tree.get_node(start_node_id)
        max_time = -1
        prob = {}
        for child_id in tmp_node.fpointer:
            child_node = sub_tree.tree.get_node(child_id)
            prob[child_node.data.word[-1]] = child_node.data.num / sub_tree.count
            if child_node.data.num > max_time:
                max_time = child_node.data.num
                select_word = child_node.data.word[-1]
                select_word_node_id = child_node.identifier
        return prob, select_word, select_word_node_id

    def _cal_policy(self, words_list, l_passage, feeddict, indx=0):
        max_id = float('-inf')
        policy_c_id = []
        words_list = map(eval, words_list)
        fd_words_list, fd_policy_c_id = [], []
        for word in words_list:
            fd_words_list.append([indx, word])
        for can in words_list:
            max_id = max(can, max_id)
        for idx in range(l_passage):
            if idx > max_id:
                policy_c_id.append(idx)
        for word in policy_c_id:
            fd_policy_c_id.append([indx, word])
        # print ('fd_policy_c_id',len(fd_policy_c_id))
        if len(words_list) == 0:
            p_pred = self.sess.run(self.prob_first, feed_dict=feeddict)
        else:
            feed_dict = dict(
                {self.selected_id_list: fd_words_list, self.candidate_id: fd_policy_c_id}.items() + feeddict.items())
            print feed_dict
            p_pred = self.sess.run(self.prob, feed_dict=feed_dict)

        return policy_c_id, p_pred

    def _policy_padding(self, padding_list, seq_length_list):
        padding_length = 0
        for length in seq_length_list:
            padding_length = max(padding_length, length)
        # print('padding_length',padding_length)
        for idx, sub_list in enumerate(padding_list, 0):
            # you yade gaixi
            # print ('len(sub_list)', len(sub_list))
            padding = [sub_list[-1][0], (sub_list[-1][1] + 1)]
            # print ('padding',padding)
            rangee = padding_length - seq_length_list[idx]
            print rangee
            for i in range(rangee):
                sub_list.append(padding)
        for sub_list in padding_list:
            assert len(sub_list) == padding_length
        return padding_list

    def _cal_policys(self, words_list_list, l_passage_list, feeddict):

        policy_c_id_list = []
        fd_words_list = []
        seq_length_list = []
        candidate_length_list = []
        fd_policy_c_id_list = []

        for idx, words_list in enumerate(words_list_list, 0):

            max_id = float('-inf')
            policy_c_id = []
            words_list = map(eval, words_list)
            tmp = []
            for word in words_list:
                tmp.append([idx, word])
            fd_words_list.append(tmp)
            seq_length_list.append(len(words_list))
            for can in words_list:
                max_id = max(can, max_id)
            for i in range(l_passage_list[idx]):
                if i > max_id:
                    policy_c_id.append(i)
            candidate_length_list.append(len(policy_c_id))
            policy_c_id_list.append(policy_c_id)
            tmp2 = []
            for word in policy_c_id:
                tmp2.append([idx, word])
            fd_policy_c_id_list.append(tmp2)
        print ('start_padding', candidate_length_list)
        fd_policy_c_id_list = self._policy_padding(fd_policy_c_id_list, candidate_length_list)
        print ('fd_words_list', fd_words_list)
        print ('shape', np.shape(fd_words_list))
        print ('fd_policy_c_id_list', fd_policy_c_id_list)
        print ('shape', np.shape(fd_policy_c_id_list))

        selected_batch_size = len(fd_words_list)
        candidate_batch_size = [len(fd_policy_c_id_list), 1, 1]
        feed_dict = dict(
            {self.selected_id_list: fd_words_list, self.candidate_id: fd_policy_c_id_list,
             self.selected_batch_size: selected_batch_size, self.candidate_batch_size: candidate_batch_size}.items()
            + feeddict.items())
        # print feed_dict
        print
        # shape_a, shape_b = self.sess.run([self.shape_a,self.shape_b],feed_dict = feed_dict)
        # print ('shape_a',shape_a)
        # print ('shape_b',shape_b)
        can = self.sess.run(self.can, feed_dict=feed_dict)
        print ('can', can)
        print ('shape of can ', np.shape(can))
        c_pred = can
        # c_pred = self.sess.run(self.prob, feed_dict=feed_dict)

        return policy_c_id_list, c_pred

    def _cal_value(self, words_list, feeddict, indx=0):
        words_list = map(eval, words_list)
        fd_words_list = []
        for word in words_list:
            fd_words_list.append([indx, word])

        if len(words_list) == 0:
            value_p = self.sess.run(self.value_first, feed_dict=feeddict)
        else:
            feed_dict = dict({self.selected_id_list: fd_words_list}.items() + feeddict.items())
            value_p = self.sess.run(self.value, feed_dict=feed_dict)

        return value_p

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
        # self._compute_loss()
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
        self.dropout_keep_prob = tf.placeholder(tf.float32)

        # test
        # self.p_words_id = tf.placeholder(tf.int32, [None,None])
        self.candidate_id = tf.placeholder(tf.int32, None)
        self.seq_length = tf.placeholder(tf.int32, [None])
        self.selected_batch_size = tf.placeholder(tf.int32, None)
        self.candidate_batch_size = tf.placeholder(tf.int32, None)
        # self.words = tf.placeholder(tf.float32, [None, None])
        self.selected_id_list = tf.placeholder(tf.int32, None)
        self.policy = tf.placeholder(tf.float32, [1, None])  # policy
        self.v = tf.placeholder(tf.float32, [1, 1])  # value

    def _embed(self):
        """
        The embedding layer, question and passage share embeddings
        """
        # with tf.device('/cpu:0'), tf.variable_scope('word_embedding'):
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
            _, self.sep_q_encodes = rnn('bi-lstm', self.q_emb, self.q_length, self.hidden_size)
            # self.sep_q_encodes,_ = rnn('bi-lstm', self.q_emb, self.q_length, self.hidden_size)
        if self.use_dropout:
            self.p_encodes = tf.nn.dropout(self.p_encodes, self.dropout_keep_prob)
            self.sep_q_encodes = tf.nn.dropout(self.sep_q_encodes, self.dropout_keep_prob)

    def _initstate(self):
        self.V = tf.Variable(
            tf.random_uniform([self.hidden_size * 2, self.hidden_size * 2], -1. / self.hidden_size,
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

        self.words = tf.reshape(self.p_encodes, [-1, self.hidden_size * 2])

    def _action_frist(self):
        """
        select first word
        """
        # self.candidate = tf.reshape(self.p_emb,[-1,self.hidden_size*2])
        self.VV = tf.expand_dims(self.V, 0)
        self.w = tf.matmul(self.p_encodes, self.VV)
        self.t_q_state = tf.expand_dims(tf.transpose(self.q_state), 0)
        ################# this place if it is duo jincheng gaizenme zuo #############
        self.tmp = tf.matmul(self.w, self.t_q_state)
        self.logits_first = tf.reshape(self.tmp, [-1])
        self.prob_first = tf.nn.softmax(self.logits_first)
        self.prob_id_first = tf.argmax(self.prob_first)
        self.value_first = tf.sigmoid(tf.reshape(tf.matmul(self.q_state, self.W), [1, 1]) + self.W_b)  # [1,1]

    def _action(self):
        """
        Employs Bi-LSTM again to fuse the context information after match layer
        """
        # self.selected_id_list = tf.expand_dims(self.selected_id_list, 0)


        self.candidate = tf.gather_nd(self.p_encodes, self.candidate_id)

        self.shape_a = tf.shape(self.seq_length)
        self.selected_list = tf.gather_nd(self.p_encodes, self.selected_id_list)

        self.rnn_input = tf.reshape(self.selected_list, [self.selected_batch_size, -1, self.hidden_size * 2])

        rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_size, state_is_tuple=False)
        _, self.states = tf.nn.dynamic_rnn(rnn_cell, self.rnn_input, sequence_length=self.seq_length,
                                           initial_state=self.q_state, dtype=tf.float32)  # [1, dim]

        self.value = tf.sigmoid(tf.matmul(self.states, self.W) + self.W_b)  # [1,1]

        # self.value = tf.sigmoid(tf.reshape(tf.matmul(self.states, self.W), [1, 1]) + self.W_b)  # [1,1]

        self.VVV = tf.tile(self.VV, self.candidate_batch_size)
        self.can = tf.matmul(self.candidate, self.VVV)

        # self.shape_a = tf.shape(self.can)
        # self.shape_b = tf.shape(self.states)
        # self.logits = tf.reshape(tf.matmul(tf.matmul(self.candidate, self.V), tf.transpose(self.states)), [-1])
        #
        # self.prob = tf.nn.softmax(self.logits)
        # self.prob_id = tf.argmax(self.prob)

    def _compute_loss(self):
        """
        The loss function
        """
        self.loss_first = tf.contrib.losses.mean_squared_error(self.v, self.value_first) - \
                          tf.matmul(self.policy,
                                    tf.reshape(tf.log(tf.clip_by_value(self.prob_first, 1e-30, 1.0)), [-1, 1]))
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


if __name__ == '__main__':
    1 == 1
    # tree_search()
    test_tf()
    # tree_search()