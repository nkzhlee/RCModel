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
    return x*x

def test_tf():
    1==1
    # x = tf.placeholder(tf.float64, shape = None)
    # y = tf.placeholder(tf.float64, shape = None)
    # z = tf.placeholder(tf.float64, shape=None)
    # a = np.ones((1,5,4))
    # b = np.array([[[1,2],[1,3]], [[0,1],[0,2]]])
    # c = np.array([[(1.,2.,3.),(2.,3.,4.),(3.,4.,5.),(4.,5.,6.)],[(1.,2.,3.),(2.,2.,2.),(3.,4.,5.),(4.,5.,6.)]])
    # #print a
    # print b
    # print c
    # print type(b)
    # #y = tf.multiply(x,)
    # tmp = tf.expand_dims(z,0)
    # sa = tf.shape(x)
    # sb = tf.shape(y)
    # sc = tf.shape(z)
    # s = tf.shape(tmp)
    #
    # #q = tf.matmul(x, tmp)
    # #sd = tf.shape(q)
    #
    # r = tf.gather_nd(c,b)
    # sr = tf.shape(r)
    # #print np.shape(a)
    # #print np.shape(b)
    # with tf.Session() as sess:
    #     sb,sc,s,tmp,r,sr= sess.run([sb,sc,s,tmp,r,sr], feed_dict={x:a,y:b,z:c})
    # print sb
    # print sc
    # #print q
    # print r
    # print sr
    # #return result


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

        self.p_word_id, self.p_pred = [],[]

        self.tmp_node = None
        self.expand_node = None
        self.num_of_search = 0

        self.result_value = 0


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
        self.evluation_m = 'Rouge-L' #'Bleu-1','Bleu-2,3,4'

        # length limit
        self.max_p_num = args.max_p_num
        self.max_p_len = args.max_p_len
        self.max_q_len = args.max_q_len

        # self.max_a_len = args.max_a_len
        self.max_a_len = 3
        # test paras
        self.search_time = 4
        self.beta = 100.0

        self._build_graph()

    def _init_sub_tree(self,tree):
        print '------- init sub tree :' + str(tree['tree_id']) + '---------'
        start_node = 'question_' + str(tree['tree_id'])
        mcts_tree = sub_tree(tree)
        data_tree = Data_tree(mcts_tree, start_node)
        data_tree.num_of_search += 1
        return data_tree

    def _do_init_tree_job(self, lock,trees_to_accomplish, trees_that_are_done, log):
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
                #result = self._init_sub_tree(tree)
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
                #time.sleep(.5)
        return True

    def _search_sub_tree(self, data_tree):
        sub_tree = data_tree.tree
        #print '------- search sub tree :' + str(sub_tree.q_id) + '---------'
        start_node_id = data_tree.start_node
        data_tree.num_of_search +=1
        data_tree.select_list=[start_node_id]
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
                lock.acquire()
                try:
                    data_tree = trees_to_accomplish.get_nowait()
                    #print ('_do_search_tree_job', type(data_tree))
                finally:
                    lock.release()
            except Queue.Empty:
                break
            else:
                '''
                    if no exception has been raised, add the task completion 
                    message to task_that_are_done queue
                '''
                #result = self._search_sub_tree(tree)
                sub_tree = data_tree.tree
                #print '------- search sub tree :' + str(sub_tree.q_id) + '---------'
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
                    trees_that_are_done.put(data_tree)
                finally:
                    lock.release()

        return True

    def _do_tree_action_job(self, lock,trees_to_accomplish, action_result_queue, log):
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
                #result = self._aciton_tree(tree)
                #result = tree
                prob, select_word_id, start_node = self._take_action(data_tree)
                data_tree.start_node = start_node
                data_tree.p_data.append(prob)
                data_tree.listSelectedSet.append(select_word_id)
                lock.acquire()
                try:
                    log.put(str(data_tree.listSelectedSet) + ' is list of action choosen by ' + str(mp.current_process().name))
                    action_result_queue.put(data_tree)
                finally:
                    lock.release()
        return True

    def feed_in_batch(self, tree_batch, parallel_size,feed_dict):
        self.tree_batch = tree_batch
        self.para_size = parallel_size
        self.batch_size = len(self.tree_batch['tree_ids'])
        #self.feed_dict = feed_dict

    def tree_search(self):
        trees = []
        #test_tf()
        time_tree_start = time.time()
        #1)initialize trees
        for bitx in range(self.batch_size):
            #print '-------------- yeild ' + str(bitx) + '-------------'
            if self.tree_batch['p_length'][bitx] > self.max_p_len:
                #print '>>>>>>>>>>>>>>>> '
                self.tree_batch['p_length'][bitx] = self.max_p_len
                self.tree_batch['candidates'][bitx] = self.tree_batch['candidates'][bitx][:(self.max_p_len)] #???
            tree = {'tree_id': self.tree_batch['tree_ids'][bitx],
                    'question_token_ids': self.tree_batch['root_tokens'][bitx],
                    'passage_token_id': self.tree_batch['candidates'][bitx],
                    'q_length': self.tree_batch['q_length'][bitx],
                    'p_length': self.tree_batch['p_length'][bitx],
                    'question_type': self.tree_batch['question_type'][bitx],
                    'ref_answer': self.tree_batch['ref_answers'][bitx]
                    #'mcst_model':self.tree_batch['mcst_model']
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
            p = mp.Process(target=self._do_init_tree_job, args=(lock,trees_to_accomplish, trees_that_are_done,log))
            processes.append(p)
            p.start()

        # completing process
        for p in processes:
            p.join()
        # while not log.empty():
        #     1==1
        #     print(log.get())

        # for i,p in enumerate(processes):
        #     if not p.is_alive():
        #         print ("[MAIN]: WORKER is a goner", i)

        # init the root node and expand the root node

        self.tree_list = []
        self.finished_tree = []
        init_list = []
        while not trees_that_are_done.empty():
            now_tree = trees_that_are_done.get()
            now_tree.expand_node = now_tree.tree.tree.get_node(now_tree.tree.tree.root)
            init_list.append(now_tree)

        self.tree_list = self.expands(init_list)
        # search tree

        for t in xrange(self.max_a_len):
            print ('Answer_len', t)
            if len(self.tree_list) == 0:
                break
            for data_tree in self.tree_list:
                has_visit_num = 0.0
                tmp_node = data_tree.tree.tree.get_node(data_tree.start_node)
                for child_id in tmp_node.fpointer:
                    child_node = data_tree.tree.tree.get_node(child_id)
                    has_visit_num += child_node.data.num
                data_tree.tree.count = has_visit_num
            #search_time =int(self.search_time- has_visit_num)
            for s_time in range(self.search_time):
                print ('search time', s_time)

                # creating processes
                processes_search = []

                tree_search_queue = manager.Queue()
                tree_result_queue = manager.Queue()

                for tree in self.tree_list:
                    #print ('type', type(tree))
                    tree_search_queue.put(tree)

                search_tree_list = []

                for w in range(number_of_procs):
                    p = mp.Process(target=self._do_search_tree_job, args=(lock, tree_search_queue, tree_result_queue, log))
                    processes_search.append(p)
                    p.start()
                    time.sleep(0.1)

                while 1:
                    if not tree_result_queue.empty():
                        data_tree = tree_result_queue.get()
                        search_tree_list.append(data_tree)
                    if len(search_tree_list) == number_of_procs:
                        break
                    #time.sleep(0.1)

                # completing process
                for p in processes_search:
                    #p.join()
                    p.terminate()

                # while not log.empty():
                #     1==1
                #     print(log.get())

                self.tree_list = []
                #gather train data

                self.tree_list = self._search_vv(search_tree_list)

                tree_need_expand_list = []
                tree_no_need_expand_list = []
                for data_tree in self.tree_list:
                    data_tree_update = self._updates(data_tree)
                    tmp_node = data_tree_update.tmp_node
                    l_passage = data_tree_update.l_passage #???
                    word_id = int(tmp_node.data.word[-1])
                    if tmp_node.is_leaf() and (word_id < (l_passage)):
                        data_tree_update.expand_node = tmp_node
                        tree_need_expand_list.append(data_tree_update)
                    else:
                        tree_no_need_expand_list.append(data_tree_update)

                self.tree_list = self.expands(tree_need_expand_list)
                self.tree_list = self.tree_list + tree_no_need_expand_list


            print '%%%%%%%%%%%%%%%%%%% start take action %%%%%%%%%%%%%%'
            num_action_procs = 0
            self.finished_tree = []
            action_queue = manager.Queue()
            action_result_queue = manager.Queue()
            for data_tree in self.tree_list:
                #print ('######### tree.listSelectedSet: ', data_tree.listSelectedSet)
                if not len(data_tree.listSelectedSet) == 0 :
                    last_word = data_tree.listSelectedSet[-1]
                    if not last_word == str(data_tree.l_passage):
                        action_queue.put(data_tree)
                        num_action_procs +=1
                    else:
                        self.finished_tree.append(data_tree)
                else:
                    action_queue.put(data_tree)
                    num_action_procs += 1
            action_tree_list = []
            processes_action = []
            #print ('###start take action ')
            #print ('len(self.tree_list)', len(self.tree_list))

            for w in range(num_action_procs):
                #print (w, w)
                p = mp.Process(target=self._do_tree_action_job, args=(lock, action_queue, action_result_queue, log))
                processes_action.append(p)
                p.start()
                time.sleep(0.1)
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

            # while not log.empty():
            #     print(log.get())

            self.tree_list = action_tree_list
            for selection in action_tree_list:
                print ('selection', selection.listSelectedSet)
            print '%%%%%%%%%%%%%% end take action %%%%%%%%%%%%%%%'

        for t in self.tree_list:
            self.finished_tree.append(t)
        time_tree_end = time.time()
        print ('&&&&&&&&&&&&&&& tree search time = %3.2f s &&&&&&&&&&&&' %(time_tree_end-time_tree_start))
        print ('--------------- end tree:', len(self.finished_tree))
        #create nodes --->search  until finish ----
        pred_answers,ref_answers = [],[]
        for data_tree in self.finished_tree:
            p_words_list = data_tree.words_id_list
            listSelectedSet_words = []
            listSelectedSet = map(eval, data_tree.listSelectedSet)
            for idx in listSelectedSet:
                listSelectedSet_words.append(p_words_list[idx])
            strr123 = self.vocab.recover_from_ids(listSelectedSet_words, 0)
            pred_answers.append({'question_id': data_tree.q_id,
                                 'question_type': data_tree.q_type,
                                 'answers': [''.join(strr123)],
                                 'entity_answers': [[]],
                                 'yesno_answers': []})
            ref_answers.append({'question_id': data_tree.q_id,
                             'question_type': data_tree.q_type,
                             'answers': data_tree.ref_answer,
                             'entity_answers': [[]],
                             'yesno_answers': []})
            if len(ref_answers) > 0:
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
            data_tree.result_value = value_with_mcts

        print '============= start compute loss ==================='
        loss_time_start = time.time()
        first_sample_list = []
        sample_list = []

        #save lists of feed_dict
        p_list,q_list = [], []
        p_length, q_length= [], []
        p_data_list = []
        pad = 0
        # selected_words_list = []
        # candidate_words_list = []
        for t_id, data_tree in enumerate(self.finished_tree,0):
            tree_data = data_tree.tree.get_raw_tree_data()
            listSelectedSet = data_tree.listSelectedSet
            pad = [t_id, len(tree_data['passage_token_id'])-1]
            #print ('pad',pad)

            words_list = [i for i in range(data_tree.l_passage+1)]
            for prob_id, prob_data in enumerate(data_tree.p_data):
                # print 'p_data: '
                # print prob_id
                # print prob_data
                c = []
                policy = []
                for prob_key, prob_value in prob_data.items():
                    c.append(prob_key)
                    policy.append(prob_value)
                # print ('tree_id', data_tree.q_id)
                # print 'listSelectedSet[:prob_id]'
                # print listSelectedSet[:prob_id]
                # print 'policy: '
                # print policy
                # print 'sum_policy'
                # print np.sum(policy)
                # print 'shape_policy'
                # print np.shape(policy)
                # print 'value: '
                # print data_tree.result_value['Rouge-L']
                # print 'candidate: '
                # print c
                if prob_id == 0:
                    input_v = data_tree.result_value['Rouge-L']
                    feed_dict = {self.p: [tree_data['passage_token_id']],
                                 self.q: [tree_data['question_token_ids']],
                                 self.p_length: [tree_data['p_length']],
                                 self.q_length: [tree_data['q_length']],
                                 self.words_list: words_list,
                                 self.dropout_keep_prob: 1.0}
                    feeddict = dict(feed_dict.items() + {self.policy: [policy], self.v: [[input_v]]}.items())

                    first_sample_list.append(feeddict)

                    _,loss_first = self.sess.run([self.optimizer_first,self.loss_first], feed_dict=feeddict)
                    print('loss,first', loss_first)
                else:
                    p_list.append(tree_data['passage_token_id'])
                    q_list.append(tree_data['question_token_ids'])
                    p_length.append(tree_data['p_length'])
                    q_length.append(tree_data['q_length'])
                    p_data_list.append([t_id,listSelectedSet[:prob_id], c, policy, data_tree.result_value[self.evluation_m]])



            # for sample in first_sample_list:

            #     loss_first = self.sess.run(self.loss_first, feed_dict=sample)
            #     print('loss,first', loss_first)
            # for sample in sample_list:
        policy_c_id_list = []
        fd_selected_list = []
        selected_length_list = []
        candidate_length_list = []
        fd_policy_c_id_list = []
        policy_list = []
        value_list = []
        for idx, sample in enumerate(p_data_list, 0):
            #print ('sample', sample)
            t_id = sample[0]
            selected_words = sample[1]
            candidate_words = sample[2]
            policy = sample[3]
            value = sample[4]

            selected_words = map(eval, selected_words)
            tmp = []
            for word in selected_words:
                tmp.append([t_id, word])
            fd_selected_list.append(tmp)
            selected_length_list.append(len(selected_words))

            candidate_words = map(eval, candidate_words)
            tmp2 = []
            for word2 in candidate_words:
                tmp2.append([t_id, word2])
            fd_policy_c_id_list.append(tmp2)
            # no order version
            candidate_length_list.append(len(candidate_words))
            assert len(candidate_words) == len(policy)
            policy_list.append(policy)
            value_list.append(value)
        fd_selected_list = self._pv_padding(fd_selected_list, selected_length_list, pad)
        fd_policy_c_id_list = self._pv_padding(fd_policy_c_id_list, candidate_length_list, pad)
        policy_list = self._pv_padding(policy_list, candidate_length_list, 0.0)

        if not (len(policy_list)) == 0:
            feed_dict = {self.p: p_list,
                              self.q: q_list,
                              self.p_length: p_length,
                              self.q_length: q_length,
                              self.dropout_keep_prob: 1.0}
        feeddict = dict(feed_dict.items() + {
            self.selected_id_list: fd_selected_list, self.seq_length:selected_length_list, self.selected_batch_size : len(selected_length_list),
            self.candidate_id: fd_policy_c_id_list, self.candidate_batch_size: [len(fd_policy_c_id_list),1,1],
            self.policy: policy_list, self.v: [value_list]}.items())
        # print ('shape of p_list',np.shape(p_list))
        # print ('shape of q_list', np.shape(q_list))
        # print ('shape of p_length', np.shape(p_length))
        # print ('shape of q_length', np.shape(q_length))
        # print ('shape of fd_selected_list', np.shape(fd_selected_list))
        # print ('shape of selected_length_list', np.shape(selected_length_list))
        # print ('shape of selected_batch_size', np.shape(len(selected_length_list)))
        # print ('shape of fd_policy_c_id_list', np.shape(fd_policy_c_id_list))
        # print ('shape of candidate_batch_size', np.shape([len(fd_policy_c_id_list),1,1]))
        # print ('shape of policy_list', np.shape(policy_list))
        # print ('shape of [value_list]', np.shape([value_list]))
        #print ('shape of ', np.shape())

        _, loss = self.sess.run([self.optimizer,self.loss], feed_dict=feeddict)
        loss_time_end = time.time()
        print('loss',loss)
        print ('time of computer loss is %3.2f s' %(loss_time_end-loss_time_start))
        print '==================== end computer loss ================ '
        # loss = self.sess.run(self.loss, feed_dict=feeddict)
        # print('loss',loss)

        # total_loss += loss * len(self.finished_tree)
        # total_num += len(self.finished_tree)
        # n_batch_loss += loss
        # if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
        #     self.logger.info('Average loss from batch {} to {} is {}'.format(
        #         bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch))
        #return 1.0 * total_loss / total_num
        


        return 0


    def _pv_padding(self, padding_list, seq_length_list, pad):
        padding_length = 0
        for length in seq_length_list:
            padding_length = max(padding_length,length)
        #print('padding_length',padding_length)
        for idx, sub_list in enumerate(padding_list,0):
            #you yade gaixi
            #print ('sublist [-1]', sub_list[-1])
            rangee = padding_length - seq_length_list[idx]
            for i in range(rangee):
                sub_list.append(pad)
        for sub_list in padding_list:
            assert len(sub_list) == padding_length
        return padding_list

    def expands(self, tree_list):
        print ('============= start expands ==============')
        time_expend_start = time.time()
        p_feed = []
        q_feed = []
        p_lenth_feed = []
        q_length_feed = []
        words_list_list = []
        l_passage_list = []
        policy_need_list = []
        for t_idx, data_tree in enumerate(tree_list,0):
            tree_data = data_tree.tree.get_raw_tree_data()
            word_list = data_tree.expand_node.data.word
            l_passage = data_tree.l_passage
            #print ('1word_list', word_list)
            if(len(word_list) == 0):
                data_tree = self._get_init_policy(data_tree,l_passage+1)
            else:
                p_feed.append(tree_data['passage_token_id'])
                q_feed.append(tree_data['question_token_ids'])
                p_lenth_feed.append(tree_data['p_length'])
                q_length_feed.append(tree_data['q_length'])
                words_list_list.append(data_tree.expand_node.data.word)
                l_passage_list.append((data_tree.l_passage+1))
                policy_need_list.append(t_idx)

        if not (len(p_feed) == 0):
            feed_dict = {self.p: p_feed,
                                  self.q: q_feed,
                                  self.p_length: p_lenth_feed,
                                  self.q_length: q_length_feed,
                                  self.dropout_keep_prob: 1.0}
            policy_ids, policys = self._cal_policys(words_list_list,l_passage_list,feed_dict)
            for p_idx, t_idx in enumerate(policy_need_list, 0):
                tree_list[t_idx].p_pred = policys[p_idx]
                tree_list[t_idx].p_word_id = policy_ids[p_idx]
        for d_tree in tree_list:
            leaf_node = d_tree.expand_node
            words_list = leaf_node.data.word
            #print ('words_list', words_list)
            for idx, word in enumerate(d_tree.p_word_id,0):
                #print ('word ', word)
                d_tree.tree.node_map[' '.join(words_list + [str(word)])] = len(d_tree.tree.node_map)
                #print ('node_map', d_tree.tree.node_map)
                new_node = node()
                new_node.word = words_list + [str(word)]
                #idx = d_tree.p_word_id.index(word)
                new_node.p = d_tree.p_pred[idx]
                # print 'new_node.p ' + str(new_node.p)
                id = d_tree.tree.node_map[' '.join(new_node.word)]
                #print 'identifier******************* ' + str(id)
                d_tree.tree.tree.create_node(identifier= id , data=new_node,
                                      parent=leaf_node.identifier)
        time_expand_end = time.time()
        print ('time of expand is %3.2f s' %(time_expand_end-time_expend_start))
        print ('================= end expands ==============')
        return tree_list

    def _get_init_policy(self, data_tree, l_passage):
        #print('&&&&&&&&& start init_policy &&&&&&&&')
        tree = data_tree.tree
        tree_data = tree.get_raw_tree_data()
        words_list = [i for i in range(l_passage)]
        feed_dict = {self.p: [tree_data['passage_token_id']],
                     self.q: [tree_data['question_token_ids']],
                     self.p_length: [tree_data['p_length']],
                     self.q_length: [tree_data['q_length']],
                     self.words_list: words_list,
                     self.dropout_keep_prob: 1.0}
        # print ('length of passage', tree_data['p_length'])
        # print ('length of padding passage',len(tree_data['passage_token_id']))
        # print ('padding',tree_data['passage_token_id'][-1])

        data_tree.p_pred = self.sess.run(self.prob_first, feed_dict=feed_dict)
        data_tree.p_word_id = [i for i in range(l_passage)]
        #print('&&&&&&&&& end init_policy &&&&&&&&')
        return data_tree

    def _get_init_value(self, data_tree):

        #print('$$$$$$$ start init_value $$$$$$$$$')
        tree = data_tree.tree
        tree_data = tree.get_raw_tree_data()
        feed_dict = {self.p: [tree_data['passage_token_id']],
                     self.q: [tree_data['question_token_ids']],
                     self.p_length: [tree_data['p_length']],
                     self.q_length: [tree_data['q_length']],
                     self.dropout_keep_prob: 1.0}

        value_p = self.sess.run(self.value_first, feed_dict=feed_dict)
        # print ('_get_init_value',value_p)
        # print('$$$$$$$ end init_value $$$$$$$$$')
        return value_p

    def _search_vv(self, search_tree_list):
        start = time.time()
        print ('--------------------- start search_vv  ------------------------')
        value_id_list = []
        p_feed = []
        q_feed = []
        p_lenth_feed = []
        q_length_feed = []
        words_list_list = []
        for t_id,data_tree in enumerate(search_tree_list,0):
            tree_data = data_tree.tree.get_raw_tree_data()
            tmp_node = data_tree.tmp_node
            word_id = int(tmp_node.data.word[-1])
            l_passage = data_tree.l_passage  ##???
            words_list = tmp_node.data.word
            if len(words_list) == 0:
                data_tree.value = self._get_init_value(data_tree)
            else:
                #print ('word_id', word_id)
                if (word_id == (l_passage)):
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
                    print '**************** tree_search get end id ***************'
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
                    v = bleu_rouge[self.evluation_m]
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

        for t_idx,v_idx in enumerate(value_id_list, 0):
            search_tree_list[v_idx].value = values[t_idx]
        end = time.time()
        print ('search time: %3.2f s' %(end - start))
        print('----------------- end search_vv ' + str(end) + '------------------')

        return search_tree_list

    def _cal_values(self, words_list_list, feeddict):
        fd_words_list = []
        seq_length = []
        for idx, words_list in enumerate(words_list_list,0):
            words_list = map(eval, words_list)
            tp = []
            for word in words_list:
                tp = np.array([idx,word])
            fd_words_list.append(tp)
            seq_length.append(np.array(len(words_list)))

        fd_words_list = np.array(fd_words_list)
        seq_length = np.array(seq_length)


        feed_dict = dict({self.selected_id_list: fd_words_list, self.seq_length: seq_length, 
                          self.selected_batch_size : len(seq_length)}.items() + feeddict.items())
        values = self.sess.run(self.value, feed_dict=feed_dict)



        #print ('values',values)
        return values

    def _updates(self,data_tree):
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

    def _policy_padding(self, padding_list, seq_length_list, pad):
        padding_length = 0
        for length in seq_length_list:
            padding_length = max(padding_length,length)
        #print('padding_length',padding_length)
        for idx, sub_list in enumerate(padding_list,0):
            #you yade gaixi
            #print ('sublist [-1]', sub_list[-1])
            #padding = [sub_list[-1][0],(sub_list[-1][1])]
            #print ('padding',padding)
            rangee = padding_length - seq_length_list[idx]
            for i in range(rangee):
                sub_list.append(pad)
        for sub_list in padding_list:
            assert len(sub_list) == padding_length
        return padding_list

    def _cal_policys(self, words_list_list, l_passage_list, feeddict):

        policy_c_id_list = []
        fd_words_list = []
        seq_length_list = []
        candidate_length_list = []
        fd_policy_c_id_list = []

        for idx, words_list in enumerate(words_list_list,0):

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
            pad = [idx,l_passage_list[idx]-1]
            candidate_length_list.append(len(policy_c_id))
            policy_c_id_list.append(policy_c_id)
            tmp2 = []
            for word in policy_c_id:
                tmp2.append([idx, word])
            fd_policy_c_id_list.append(tmp2)
        #print ('start_padding', candidate_length_list)
        fd_policy_c_id_list = self._policy_padding(fd_policy_c_id_list,candidate_length_list,pad)

        selected_batch_size = len(fd_words_list)
        candidate_batch_size  = [len(fd_policy_c_id_list),1,1]
        feed_dict = dict(
            {self.selected_id_list: fd_words_list, self.candidate_id: fd_policy_c_id_list, self.seq_length: seq_length_list,
             self.selected_batch_size : selected_batch_size, self.candidate_batch_size: candidate_batch_size}.items()
            + feeddict.items())
        #print feed_dict
        c_pred = self.sess.run(self.prob, feed_dict=feed_dict)
        #print ('can', c_pred)
        #print ('shape of pre ', np.shape(c_pred))
        # for x in c_pred:
        #     print ('x',np.sum(x))
        #c_pred = self.sess.run(self.prob, feed_dict=feed_dict)

        return policy_c_id_list, c_pred

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
        self.dropout_keep_prob = tf.placeholder(tf.float32)

        # test
        self.words_list = tf.placeholder(tf.int32, [None])
        self.candidate_id = tf.placeholder(tf.int32, None)
        self.seq_length =  tf.placeholder(tf.int32, [None])
        self.selected_batch_size = tf.placeholder(tf.int32,None)
        self.candidate_batch_size = tf.placeholder(tf.int32, None)
        # self.words = tf.placeholder(tf.float32, [None, None])
        self.selected_id_list = tf.placeholder(tf.int32, None)
        self.policy = tf.placeholder(tf.float32, [None, None])  # policy
        self.v = tf.placeholder(tf.float32, [1,None])  # value


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
            #self.sep_q_encodes,_ = rnn('bi-lstm', self.q_emb, self.q_length, self.hidden_size)
        if self.use_dropout:
            self.p_encodes = tf.nn.dropout(self.p_encodes, self.dropout_keep_prob)
            self.sep_q_encodes = tf.nn.dropout(self.sep_q_encodes, self.dropout_keep_prob)


    def _initstate(self):
        self.V = tf.Variable(
            tf.random_uniform([self.hidden_size * 2, self.hidden_size * 2], -1. / self.hidden_size, 1. / self.hidden_size))
        self.W = tf.Variable(tf.random_uniform([self.hidden_size * 2, 1], -1. / self.hidden_size, 1. / self.hidden_size))

        self.W_b = tf.Variable(tf.random_uniform([1, 1], -1. / self.hidden_size, 1. / self.hidden_size))

        self.V_c = tf.Variable(
            tf.random_uniform([self.hidden_size * 2, self.hidden_size], -1. / self.hidden_size, 1. / self.hidden_size))
        self.V_h = tf.Variable(
            tf.random_uniform([self.hidden_size * 2, self.hidden_size], -1. / self.hidden_size, 1. / self.hidden_size))

        self.q_state_c = tf.sigmoid(tf.matmul(self.sep_q_encodes, self.V_c))
        self.q_state_h = tf.sigmoid(tf.matmul(self.sep_q_encodes, self.V_h))
        self.q_state = tf.concat([self.q_state_c, self.q_state_h], 1) #(3,300)



        self.words = tf.reshape(self.p_encodes, [-1, self.hidden_size * 2])


    def _action_frist(self):
        """
        select first word
        """
        # self.candidate = tf.reshape(self.p_emb,[-1,self.hidden_size*2])
        self.words_in = tf.gather(self.words, self.words_list)

        self.w = tf.matmul(self.words_in, self.V)
        self.tmp = tf.matmul(self.w, tf.transpose(self.q_state))
        self.logits_first = tf.reshape(self.tmp, [-1])
        self.prob_first = tf.nn.softmax(self.logits_first)
        self.prob_id_first = tf.argmax(self.prob_first)
        self.value_first = tf.sigmoid(tf.reshape(tf.matmul(self.q_state, self.W), [1, 1]) + self.W_b)  # [1,1]


    def _action(self):
        """
        Employs Bi-LSTM again to fuse the context information after match layer
        """
        #self.selected_id_list = tf.expand_dims(self.selected_id_list, 0)


        self.candidate = tf.gather_nd(self.p_encodes, self.candidate_id)

        self.shape_a = tf.shape(self.seq_length)
        self.selected_list = tf.gather_nd(self.p_encodes, self.selected_id_list)


        self.rnn_input = tf.reshape(self.selected_list, [self.selected_batch_size, -1, self.hidden_size * 2]) # (6,2,300)

        #
        rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_size, state_is_tuple=False)
        _, self.states = tf.nn.dynamic_rnn(rnn_cell, self.rnn_input, sequence_length = self.seq_length , initial_state=self.q_state, dtype=tf.float32)  # [1, dim]
        #（6，300）
        self.value = tf.sigmoid(tf.matmul(self.states, self.W) + self.W_b)  # [6,1]

        #self.value = tf.sigmoid(tf.reshape(tf.matmul(self.states, self.W), [1, 1]) + self.W_b)  # [1,1]

        self.VV = tf.expand_dims(self.V, 0)

        self.VVV = tf.tile(self.VV, self.candidate_batch_size)
        self.can = tf.matmul(self.candidate, self.VVV)

        #self.s_states = tf.reshape(self.states, [self.candidate_batch_size, self.hidden_size * 2, 1])
        self.s_states = tf.expand_dims(self.states, 2)
        # self.shape_a = tf.shape(self.can)
        # self.shape_b = tf.shape(self.s_states)
        self.logits = tf.matmul(self.can, self.s_states)
        self.prob = tf.nn.softmax(self.logits,dim = 1)# (6,458,1)
        self.prob_id = tf.argmax(self.prob)



    def _compute_loss(self):
        """
        The loss function
        """
        self.loss_first = tf.contrib.losses.mean_squared_error(self.v, self.value_first) - \
                          tf.matmul(self.policy,tf.reshape(tf.log( tf.clip_by_value(self.prob_first, 1e-30,1.0)),[-1, 1]))

        self.optimizer_first = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss_first)


        self.loss = tf.reduce_mean(tf.contrib.losses.mean_squared_error(tf.transpose(self.v), self.value) - tf.reduce_sum(tf.multiply(self.policy, tf.reshape(
            tf.log(tf.clip_by_value(self.prob, 1e-30, 1.0)), [self.selected_batch_size, -1])),1))

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
    #tree_search()
    test_tf()
    #tree_search()