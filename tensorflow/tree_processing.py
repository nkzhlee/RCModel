# !/usr/bin/python
# -*- coding:utf-8 -*-

import time
import os
import numpy as np
import tensorflow as tf
from pathos.multiprocessing import ProcessPool
from layers.basic_rnn import rnn
import multiprocessing as mp
import Queue

class TFGraph(object):
    """
    Implements the main reading comprehension model.

    python -u run.py --train --algo MCST --epochs 10 --batch_size 1 --max_p_len 10000 --hidden_size 150  --train_files ../data/demo/trainset/test10 --dev_files  ../data/demo/devset/test20 --test_files ../data/demo/test/search.test.json
    """

    def __init__(self, name):
        self.tf_name = name
        self.vocab = '123'
        self.draw_path = '123'
        self.model_dir = '../data/models/'
        # basic config

        self.hidden_size = 300
        self.learning_rate = 0.1
        self.weight_decay = 1
        self.use_dropout = 1

        self.feed_dict = {}
        self._build_graph()

    def _build_graph(self):
        """
        Builds the computation graph with Tensorflow
        """
        # session info
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        self._initstate()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())


    def _initstate(self):
        """
        Placeholders
    
        """
        # self.pp = tf.placeholder(tf.float32, [None, None])
        # self.WW = tf.Variable(
        #     tf.random_uniform([1000, 1000], -1.0, 1.0))
        self.w1 = tf.placeholder("float", name="w1")
        self.w2 = tf.placeholder("float", name="w2")

        b1 = tf.Variable(2.0, name="bias")
        w3 = tf.add(self.w1, self.w2)
        self.result = tf.multiply(w3, b1, name="op_to_restore")

    def set_feed(self, w1, w2):
        self.feed_dict = {self.w1: w1, self.w2: w2}

    def save(self, model_prefix):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        #self.saver.save(self.sess, os.path.join(self.model_dir, model_prefix))

        self.saver.save(self.sess, '../data/models/test'+model_prefix)
        print 'save ../data/models/test'
        #print ('Model saved in {}, with prefix {}.'.format(self.model_dir, model_prefix))

    def restore(self, model_prefix):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        self.saver.restore(self.sess, os.path.join(self.model_dir, model_prefix))
        print('Model restored from {}, with prefix {}'.format(self.model_dir, model_prefix))


class CTFGraph(object):
    """
    Implements the main reading comprehension model.

    python -u run.py --train --algo MCST --epochs 10 --batch_size 1 --max_p_len 10000 --hidden_size 150  --train_files ../data/demo/trainset/test10 --dev_files  ../data/demo/devset/test20 --test_files ../data/demo/test/search.test.json
    """

    def __init__(self, name,model_prefix):
        self.tf_name = name
        self.vocab = '1234'
        self.draw_path = '1234'
        self.model_dir = '../data/models/test'
        self.model_prefix = model_prefix
        # basic config

        self.feed_dict = {}
        self._build_graph()

    def _build_graph(self):
        """
        Builds the computation graph with Tensorflow
        """
        # session info
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.graph = tf.get_default_graph()

        self.saver = tf.train.import_meta_graph(self.model_dir+self.model_prefix+'.meta')

        print '00'
        self.saver.restore(self.sess, self.model_dir+self.model_prefix)
        print '111'

        self._initstate()

    def _initstate(self):
        """
        Placeholders

        """
        # self.pp = tf.placeholder(tf.float32, [None, None])
        # self.WW = tf.Variable(
        #     tf.random_uniform([1000, 1000], -1.0, 1.0))

        self.w1 = self.graph.get_tensor_by_name("w1:0")
        self.w2 = self.graph.get_tensor_by_name("w2:0")

        self.result = self.graph.get_tensor_by_name("op_to_restore:0")

    def set_feed(self, w1, w2):
        self.feed_dict = {self.w1: w1, self.w2: w2}

    def save(self, model_dir, model_prefix):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        self.saver.save(self.sess, os.path.join('../data/models/', model_prefix))
        print ('Model saved in {}, with prefix {}.'.format(model_dir, model_prefix))

    def restore(self, model_prefix):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        self.saver.restore(self.sess,'../data/models/test' )
        print 'restore model ../data/models/test'
        # self.saver.restore(self.sess, os.path.join(self.model_dir, model_prefix))
        # print('Model restored from {}, with prefix {}'.format(self.model_dir, model_prefix))

def ken(list):
    graph = list[0]
    p =list[1]
    feed_dict = dict({graph.pp: p})
    v, shape = graph.sess.run([graph.V, graph.shape], feed_dict=feed_dict)
    return v,p

def ken2(list):


    w1 = list[0]
    w2 = list[1]
    graph = CTFGraph('ctf' + str(w1))
    feed_dict = dict({graph.w1: w1, graph.w2: w2})
    print ('bias: ', graph.sess.run('bias:0'))
    v = graph.sess.run(graph.result, feed_dict=feed_dict)
    return v


def ken3(lock, v_to_accomplish, main_rf_cal,v_are_done, log):
    while True:
        try:
            '''
                try to get task from the queue. get_nowait() function will 
                raise queue.Empty exception if the queue is empty. 
                queue(False) function would do the same task also.
            '''
            #with lock:
            list = v_to_accomplish.get_nowait()
        except Queue.Empty:

            break
        else:
            '''
                if no exception has been raised, add the task completion 
                message to task_that_are_done queue
            '''
            #print list

            v0 = list
            v1 = v0 + 1
            main_rf_cal.put(v1)
            tmp = 0
            while tmp == 0:
                if not v_to_accomplish.empty():
                    tmp = v_to_accomplish.get_nowait()
                    print ('tmp',tmp)
            v = (v1 + tmp) / 2
            log.put(str(v)+ ' is done by ' + str(mp.current_process().name))
            v_are_done.put(v)
            print v
    return True
#def ken4(tf):


def main():
    # p = subprocess.Popen(["python", "-u", "pmctree.py"], stdout= subprocess.PIPE, stderr = subprocess.STDOUT)
    # #p = subprocess.Popen(["ls","-l"],stdout = subprocess.PIPE)
    # output, err = p.communicate()
    # print output
    # while p.poll() == None:  # 检查子进程是否已经结束
    #     print(p.stdout.readline())
    #     time.sleep(1)
    # print(p.stdout.read())
    # print('returen code:', p.returncode)
    # session info
    graph = TFGraph('tf')
    w1 = 1.0
    w2 = 2.0
    feed_dict = dict({graph.w1: w1, graph.w2: w2})
    v = graph.sess.run(graph.result, feed_dict=feed_dict)
    print('Result is :', v)

    # input_list = [1.0]
    # out_list = []
    # for i, p in enumerate(input_list):
    #     ip = []
    #     ip.append(p)
    #     ip.append(i * 1.0)
    #     out_list.append(ip)
    # pool = ProcessPool(nodes=1)
    # results = pool.amap(ken2, out_list)
    # while not results.ready():
    #     time.sleep(0.01)
    # for value in results.get():
    #     print ('value is: ',value)
    #
    # cgraph = CTFGraph('dtf', '0')
    # w1 = 3.0
    # w2 = 4.0
    # print ('bias: ', cgraph.sess.run('bias:0'))
    # feed_dict = dict({cgraph.w1: w1, cgraph.w2: w2})
    # v = cgraph.sess.run(cgraph.result, feed_dict=feed_dict)
    # print('Cgraph Result is :', v)
    #
    # input_list = [1.0, 2.0, 3.0]
    # out_list = []
    # for i, p in enumerate(input_list):
    #     ip = []
    #     ip.append(p)
    #     ip.append(i * 1.0)
    #     ip.append(str(i))
    #     out_list.append(ip)


    result_num = 3
    init_num = [0, 1, 2]
    result_list = []
    v_to_accomplish_list = []
    main_rf_cal_list = []
    #create communitcat queen
    for i in range(result_num):
        v_to_accomplish = mp.Queue()
        v_to_accomplish.put(init_num[i])
        main_rf_cal = mp.Queue()
        v_to_accomplish_list.append(v_to_accomplish)
        main_rf_cal_list.append(main_rf_cal)


    number_of_procs = 3
    v_are_done = mp.Queue()
    log = mp.Queue()
    processes = []
    lock = mp.Lock()

    # creating processes
    for w in range(number_of_procs):
        p = mp.Process(target=ken3, args=(lock, v_to_accomplish_list[w], main_rf_cal_list[w],v_are_done, log))
        processes.append(p)
        p.start()



    while not len(result_list)==3:
        #print '1111'
        while not v_are_done.empty():
            v = v_are_done.get()
            result_list.append(v)
            print ('v', v)
        for id,cal_queue in enumerate(main_rf_cal_list,0):
            while not cal_queue.empty():
                w = cal_queue.get()
                feed_dict = dict({graph.w1: w, graph.w2: w})
                v = graph.sess.run(graph.result, feed_dict=feed_dict)
                v_to_accomplish_list[id].put(v)
    # completing process

    for p in processes:
        p.join()

    # init the root node and expand the root node





    # input_list = [1.0,2.0,3.0]
    # out_list = []
    # for i, p in enumerate(input_list):
    #     ip = []
    #     ip.append(p)
    #     ip.append(i * 1.0)
    #     out_list.append(ip)
    # pool = ProcessPool(nodes=3)
    # results = pool.amap(ken2, out_list)
    # while not results.ready():
    #     time.sleep(0.01)
    # for value in results.get():
    #     print ('value is: ',value)
    # scal = 100
    # p_list = []
    # time_start = time.time()
    # for i in range(scal):
    #     p = np.random.random((1000,1000))
    #     p_list.append(p)
    # # danji
    # for p in p_list:
    #     feed_dict = dict({graph.pp: p})
    #     v,shape = graph.sess.run([graph.V, graph.shape], feed_dict=feed_dict)
    # ave_time = (time.time() - time_start) / scal
    # print('Average time of danji calculation is :', ave_time)
    #
    # time_start = time.time()
    # input_list = []
    # for p in p_list:
    #     ip = []
    #     ip.append(graph)
    #     ip.append(p)
    #     input_list.append(ip)
    # pool = ProcessPool(nodes=5)
    # results = pool.amap(ken, p_list)
    # while not results.ready():
    #     time.sleep(0.01)
    # ave_time = (time.time() - time_start) / scal
    # print('Average time of pallal calculation is :', ave_time)


# print('has init ', data_tree.q_id)




if __name__ == '__main__':
    main()