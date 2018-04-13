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
from utils import compute_bleu_rouge
from utils import normalize

from search import SearchTree
from pmctree import PSCHTree

from tfgraph import TFGraph


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
        self.max_a_len = args.max_a_len

        self.Bleu4 = args.Bleu4
        self.Bleu3 = args.Bleu3
        self.Bleu2 = args.Bleu2
        self.Bleu1 = args.Bleu1
        self.RougeL = args.RougeL

        self.m_value = {'Bleu-4': self.Bleu4,'Bleu-3': self.Bleu3,'Bleu-2': self.Bleu2,'Bleu-1': self.Bleu1,'Rouge-L':self.RougeL}
        #test paras
        self.search_time = args.search_time
        self.beta = args.beta
        #time
        self.init_times = 0.0
        self.search_times = 0.0
        self.act_times = 0.0
        self.grad_times = 0.0


        # the vocab
        self.vocab = vocab
        #self.tfg = TFGraph('train', vocab, args)

    def _train_epoch_new(self, pmct, train_batches, batch_size, dropout_keep_prob):
        """
               Trains the model for a single epoch.
               Args:
                   train_batches: iterable batch data for training
                   dropout_keep_prob: float value indicating dropout keep probability
               """

        total_loss, num_loss = 0, 0
        for bitx, batch in enumerate(train_batches, 1):
            s_time = time.time()
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
            }
            feed_dict = {}
            pmct.feed_in_batch(tree_batch, batch_size, feed_dict, self.m_value)
            loss = pmct.tree_search(bitx)
            total_loss += loss
            num_loss += 1
            print ('use time : ', (time.time()-s_time)/60)
        #return 0
        return 1.0 * total_loss / num_loss

    # def _train_epoch_new(self, pmct, train_batches, batch_size, dropout_keep_prob):
    #     """
    #            Trains the model for a single epoch.
    #            Args:
    #                train_batches: iterable batch data for training
    #                dropout_keep_prob: float value indicating dropout keep probability
    #            """
    #
    #     total_loss, num_loss = 0, 0
    #     for bitx, batch in enumerate(train_batches, 1):
    #         print '------ Batch Question: ' + str(bitx)
    #         '''
    #         feed_dict = {self.p: batch['passage_token_ids'],
    #                           self.q: [batch['question_token_ids']],
    #                           self.p_length: batch['passage_length'],
    #                           self.q_length: [batch['question_length']],
    #                           self.dropout_keep_prob: dropout_keep_prob}
    #         '''
    #         pred_answers = {}
    #         #print str(ref_answers)
    #         listSelectedSet = []
    #         p_data = []
    #         tree_batch = {
    #             'tree_ids': batch['question_ids'],
    #             'question_type': batch['question_types'],
    #             'root_tokens': batch['question_token_ids'],
    #             'q_length': batch['question_length'],
    #             'candidates': batch['passage_token_ids'],
    #             'p_length': batch['passage_length'],
    #             'ref_answers': batch['ref_answers'],
    #         }
    #         feed_dict = {}
    #         pmct.feed_in_batch(tree_batch, batch_size, feed_dict, self.m_value)
    #         loss = pmct.tree_search()
    #         total_loss += loss
    #         num_loss += 1
    #     #return 0
    #     return 1.0 * total_loss / num_loss


    def _train_epoch(self, step , train_batches, dropout_keep_prob):
        """
        Trains the model for a single epoch.
        Args:
            train_batches: iterable batch data for training
            dropout_keep_prob: float value indicating dropout keep probability
        """
        total_loss = 0
        num_loss = 0
        batch_start_time = 0
        batch_start_time = time.time()
        for fbitx, batch in enumerate(train_batches, 1):
            step += 1
            if fbitx % 10 == 0:
                print '------ Batch Question: ' + str(fbitx)

            trees = []
            batch_tree_set = []

            batch_size = len(batch['question_ids'])
            #print ('batch_size)', batch_size)
            for bitx in range(batch_size):
                if batch['passage_length'][bitx] > self.max_p_len:
                    batch['passage_length'][bitx] = self.max_p_len
                    batch['passage_token_id'][bitx] = batch['passage_token_id'][bitx][:(self.max_p_len)]  # ???
                tree = {'question_id': batch['question_ids'][bitx],
                        'question_token_ids': batch['question_token_ids'][bitx],
                        'passage_token_ids': batch['passage_token_ids'][bitx],
                        'q_length': batch['question_length'][bitx],
                        'p_length': batch['passage_length'][bitx],
                        'question_type': batch['question_types'][bitx],
                        'ref_answers': batch['ref_answers'][bitx]
                        }
                trees.append(tree)
                #print batch
                batch_tree = SearchTree(self.tfg, tree, self.max_a_len, self.search_time, self.beta, self.m_value, dropout_keep_prob)
                batch_tree_set.append(batch_tree)


            # for every data in batch do training process
            for idx, batch_tree in enumerate(batch_tree_set,1):
                loss = batch_tree.one_train(step)
                if fbitx % 10 == 0:
                    print '++++++++++++ loss is ' + str(loss)
                num_loss += 1
                total_loss += loss


            if fbitx % 10 == 0:
                batch_end_time = time.time()
                print ('&&&&&&&&&&&&&&& batch process time = %3.2f min &&&&&&&&&&&&' % ((batch_end_time - batch_start_time)/60))
                batch_start_time = time.time()
        return 1.0 * total_loss / num_loss, step

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
        # print 'pad_id is '
        # print pad_id
        max_bleu_4 = 0
        train_step = 0
        test_step = 0
        pmct = PSCHTree(self.args, self.vocab)
        start_all_time = time.time()

        for epoch in range(1, epochs + 1):
            self.logger.info('Training the model for epoch {}'.format(epoch))
            epoch_start_time = time.time()
            train_batches = data.gen_batches('train', batch_size, pad_id, shuffle=True)
            # mctree = MCtree(train_batches)
            # mctree.search()

            result = self._train_epoch_new(pmct, train_batches, batch_size, dropout_keep_prob)
            #result, train_step = self._train_epoch(train_step, train_batches, dropout_keep_prob)
            epoch_end_time = time.time()
            self.logger.info('Average train loss for epoch {} is {}'.format(epoch, result))
            #self.save(save_dir, save_prefix + '_' + str(epoch))
            self.logger.info(
                'Train time for epoch {} is {} min'.format(epoch, str((epoch_end_time - epoch_start_time) / 60)))
            if evaluate:
                self.logger.info('Evaluating the model after epoch {}'.format(epoch))
                if data.dev_set is not None:
                    eval_loss, total_loss, num_loss = 0, 0, 0
                    eval_batches = data.gen_batches('dev', batch_size, pad_id, shuffle=False)
                    #bleu_rouge,test_step = self.evaluate(test_step, eval_batches,dropout_keep_prob)
                    # ref_answers, pre_answers = [],[]
                    # for bitx, batch in enumerate(eval_batches, 1):
                    #     print '------ Batch Question: ' + str(bitx)
                    #     #print batch
                    #     tree_batch = {
                    #         'tree_ids': batch['question_ids'],
                    #         'question_type': batch['question_types'],
                    #         'root_tokens': batch['question_token_ids'],
                    #         'q_length': batch['question_length'],
                    #         'candidates': batch['passage_token_ids'],
                    #         'p_length': batch['passage_length'],
                    #         'ref_answers': batch['ref_answers'],
                    #     }
                    #     p, r, loss = pmct.evaluate_tree_search(tree_batch)
                    #     total_loss += loss
                    #     num_loss += 1
                    #     ref_answers = ref_answers + r
                    #     pre_answers = pre_answers + p
                    # if len(ref_answers) > 0:
                    #     pred_dict, ref_dict = {}, {}
                    #     for pred, ref in zip(pre_answers, ref_answers):
                    #         question_id = ref['question_id']
                    #         if len(ref['answers']) > 0:
                    #             pred_dict[question_id] = normalize(pred['answers'])
                    #             ref_dict[question_id] = normalize(ref['answers'])
                    #             if bitx%5 == 0:
                    #                 print ('pred answer', pred_dict[question_id])
                    #                 print ('ref answer', ref_dict[question_id] )
                    #     bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
                    # #eval_loss, bleu_rouge = self.evaluate(eval_batches)
                    # self.logger.info('Dev eval loss {}'.format(1.0 * total_loss / num_loss))
                    #self.logger.info('Dev eval result: {}'.format(bleu_rouge))
            #
            #         if bleu_rouge['Bleu-4'] > max_bleu_4:
            #             pmct.save(save_dir, save_prefix)
            #             max_bleu_4 = bleu_rouge['Bleu-4']
            #     else:
            #         self.logger.warning('No dev set is loaded for evaluation in the dataset!')
            # else:
            #     pmct.save(save_dir, save_prefix + '_' + str(epoch))
        self.logger.info(
            'All Train time is {} min'.format(str((time.time() - start_all_time) / 60)))



    def evaluate(self, step, eval_batches, dropout_keep_prob,result_dir=None, result_prefix=None, save_full_info=False):
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
        for b_itx, batch in enumerate(eval_batches, 1):
            step += 1
            print '------ evaluate Batch Question: ' + str(b_itx)
            # ref_answers.append({'question_id': batch['question_ids'],
            #                     'question_type': batch['question_types'],
            #                     'answers': batch['ref_answers'],
            #                     'entity_answers': [[]],
            #                     'yesno_answers': []})

            # print 'ref_answers: '
            # print ref_answers

            trees = []
            batch_tree_set = []
            batch_start_time = time.time()
            batch_size = len(batch['question_ids'])
            # print ('batch_size)', batch_size)
            for bitx in range(batch_size):
                if batch['passage_length'][bitx] > self.max_p_len:
                    batch['passage_length'][bitx] = self.max_p_len
                    batch['passage_token_id'][bitx] = batch['passage_token_id'][bitx][:(self.max_p_len)]  # ???
                tree = {'question_id': batch['question_ids'][bitx],
                        'question_token_ids': batch['question_token_ids'][bitx],
                        'passage_token_ids': batch['passage_token_ids'][bitx],
                        'q_length': batch['question_length'][bitx],
                        'p_length': batch['passage_length'][bitx],
                        'question_type': batch['question_types'][bitx],
                        'ref_answers': batch['ref_answers'][bitx]
                        }
                trees.append(tree)
                ref_answers.append({'question_id': tree['question_id'],
                                    'question_type': tree['question_type'],
                                    'answers': tree['ref_answers'],
                                    'entity_answers': [[]],
                                    'yesno_answers': []})
                # print batch
                batch_tree = SearchTree(self.tfg, tree, self.max_a_len, self.search_time, self.beta,self.m_value, dropout_keep_prob)
                batch_tree_set.append(batch_tree)

            # for every data in batch do training process
            for idx, batch_tree in enumerate(batch_tree_set, 1):
                pred_answer, avar_loss = batch_tree.one_evaluate(step)
                pred_answers.append(pred_answer)
        # print 'ref_answers: '
        # print ref_answers

        # print 'ref_answers: '
        # print ref_answers
        print 'test avar_loss: '
        print avar_loss
        ii = 0
        if len(ref_answers) > 0:
            pred_dict, ref_dict = {}, {}
            for pred, ref in zip(pred_answers, ref_answers):
                ii += 1

                question_id = ref['question_id']
                if len(ref['answers']) > 0:
                    if( ii%2 == 0):
                        print 'pred_answers'
                        print pred['answers'][0]
                        print 'ref_answers'
                        print ref['answers'][0]

                    pred_dict[question_id] = normalize(pred['answers'])
                    ref_dict[question_id] = normalize(ref['answers'])
                    # print '========compare======='
                    # print pred_dict[question_id]
                    # print '----------------------'
                    # print ref_dict[question_id]
            bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
        else:
            bleu_rouge = None
        value_with_mcts = bleu_rouge
        print value_with_mcts
        return value_with_mcts, step


