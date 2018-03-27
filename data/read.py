# -*- coding:utf8 -*-
"""
Decode and show the json data in DuReader dataset
  python read.py ./demo/trainset/search.train.json 10
data 2018.1.9
author Zh lee
"""
import json
import sys
import logging
import argparse
import pprint


lenth_sent = 100 # >0
lenth_para = 3 # >0
lenth_doc = 2 # 0-5
f = open(sys.argv[1])
row_num = int(sys.argv[2])



def parse_args():
    f = open(sys.argv[1])
    row_num = int(sys.argv[2])
def cut_sentence(sent):
    """
        Only show sentence head -n 20
    :param sent: 
    :return: short sent
    """
    lenth = len(str(sent))
    if lenth > lenth_sent :
        sent = str(sent)[:lenth_sent]+"......"
    return sent
def read_data_set():
    """
    Read data in search_train_set
    :return: 
    """
    logger = logging.getLogger("read")
    logger.info("Read search train set")
    for i in range(0, row_num):
        print "row_num",i
        print "\033[0;30;46m-----------------------start------------------------------\033[0m"
        line = f.readline()
        data = json.loads(line)
        for key, value in data.items():
            #print "key: " + key

            if key == "question_id":
                print "\033[1;31;40m --question_id:﻿\033[0m", value
            elif key == "question_type":
                print "\033[1;31;40m --quesiton_type: \033[0m", value
            elif key == "question":
                print "\033[1;31;40m --question: \033[0m", value
            elif key == "fact_or_opinion":
                print "\033[1;32;40m --fact_or_opinion: \033[0m" , value
            elif key == "answers":
                n = 1
                for v in value:
                    ans_str = v.encode('utf-8').strip()
                    ans_str = cut_sentence(ans_str)
                    print "\033[1;36;40m ----Number "+ str(n)+ " answers: \033[0m" , ans_str
                    n = n+1
            elif key == "entity_answers":
                n = 0
                for v in value:
                    entity_str = "【"
                    for vv in v:
                        entity_str += " \""
                        entity_str += vv.encode('utf-8').strip()
                        entity_str += "\" "
                    entity_str = entity_str + "】"
                    entity_str = cut_sentence(entity_str)
                    print "\033[1;36;40m ----Number "+ str(n+1)+" entity_answers:\033[0m", entity_str
                    n = n + 1
            elif key == "yesno_answers":
                n = 0
                for v in value:
                    print "\033[1;36;40m --Number " + str(n+1) + " answer is yes_or_no:\033[0m", v
                    n = n + 1
            elif key == "documents":
                n = 0
                print "\033[1;32;40m --documents \033[0m" + "\033[1;32;40m :{\033[0m"
                for pair in value:
                    n += 1
                    if n > lenth_doc :
                        print "\033[1;36;40m    ...... \033[0m"
                        break
                    print "\033[1;36;40m    ----doc_id : \033[0m" + str(n)
                    for (k,v) in pair.items() :
                        n_para = 0
                        if k == "is_selected":
                            print "\033[1;41;40m    ----is_selected: \033[0m", v
                        elif k == "bs_rank_pos":
                            print "\033[1;41;40m    ----rank position: \033[0m", v
                        elif k == "title":
                            print "\033[1;41;40m    ----title: \033[0m", v.encode('utf-8').strip()
                        elif k== "paragraphs":
                            print "\033[1;41;40m    ----paragraphs: \033[0m"
                            if isinstance(v,list):
                                for vv in v:
                                    n_para = n_para + 1
                                    if n_para > 3 :
                                        print "\033[1;41;40m       ...... \033[0m"
                                        break
                                    if isinstance(vv, list):
                                        ss = str(vv)
                                    else: ss = cut_sentence(vv.encode('utf-8').strip())
                                    para_str = "       【" + ss + "】"
                                    print para_str
                        else: logger.info("unknown key"+ k)

                print "\033[1;32;40m }\033[0m"
            else:
                logger.info("no such key:"+str(key))
                print key
        print "\033[0;30;43m-----------------------end------------------------------\033[0m"


def run():

    parse_args()
    logger = logging.getLogger("read")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.info('Running with args : {}'.format(sys.argv))
    #logger.info("File lenth :".format(len(f.readlines())))
    read_data_set()


if __name__ == '__main__':
    run()









