# -*- coding:utf8 -*-
"""
Draw data analysis part

author zh lee
date 2018.1.16
"""

from matplotlib import pyplot as plt
import sys
import logging
import json
import numpy as np

path = "./raw/results/search.train.json.result"

#path = "./raw/results/description.search.train.json.result"
#path = "./raw/results/entity.search.train.json.result"
#path = "./raw/results/yesno.search.train.json.result"


in_doc_nwords = []
in_para_nwords = []
span_para = []
span_doc = []


def process_data(data, tmp_span):
    tmp_a = 0
    tmp_b = 0
    tmp_c = 0
    tmp_d = 0
    tmp_e = 0
    for x in data:
        if (x > 0.0) & (x <= 0.2): tmp_a += 1
        if (x > 0.2) & (x <= 0.4): tmp_b += 1
        if (x > 0.4) & (x <= 0.6): tmp_c += 1
        if (x > 0.6) & (x <= 0.8): tmp_d += 1
        if (x > 0.8) & (x <= 1.0): tmp_e += 1
    tmp_span.append(tmp_a)
    tmp_span.append(tmp_b)
    tmp_span.append(tmp_c)
    tmp_span.append(tmp_d)
    tmp_span.append(tmp_e)


def get_data():
    with open(path) as fin:
        line = fin.readline()
        while line :
            print line
            data = json.loads(line)
            for (k,v) in data.items():
                if k == "in_doc_nwords" :
                    value = float(v)
                    in_doc_nwords.append(value)
                if k == "in_para_nwords" :
                    value = float(v)
                    in_para_nwords.append(value)
            line = fin.readline()

def draw_pale():
    process_data(in_doc_nwords, span_doc)
    process_data(in_para_nwords, span_para)
    plt.figure()
    n = 5
    X = np.arange(n) + 1
    # X是1,2,3,4,5,6,7,8,柱的个数
    # numpy.random.uniform(low=0.0, high=1.0, size=None), normal
    # uniform均匀分布的随机数，normal是正态分布的随机数，0.5-1均匀分布的数，一共有n个
    #Y1 = np.random.uniform(0.5, 1.0, n)
    #Y2 = np.random.uniform(0.5, 1.0, n)
    plt.bar(X, span_doc, width=0.35, facecolor='lightskyblue', edgecolor='white')
    # width:柱的宽度
    plt.bar(X + 0.35, span_para , width=0.35, facecolor='yellowgreen', edgecolor='white')
    # 水平柱状图plt.barh，属性中宽度width变成了高度height
    # 打两组数据时用+
    # facecolor柱状图里填充的颜色
    # edgecolor是边框的颜色
    # 想把一组数据打到下边，在数据前使用负号
    # plt.bar(X, -Y2, width=width, facecolor='#ff9999', edgecolor='white')
    # 给图加text
    for x, y in zip(X, span_doc):
        #plt.text(x + 0.3, y + 0.05, '%.2f' % y, ha='center', va='bottom')
        plt.text(x + 0.3, y + 0.05, '%d' % y, ha='center', va='bottom')

    for x, y in zip(X, span_para):
        #plt.text(x + 0.6, y + 0.05, '%.2f' % y, ha='center', va='bottom')
        plt.text(x + 0.6, y + 0.05, '%d' % y, ha='center', va='bottom')
    plt.ylim(0, +2250)
    plt.show()


def run():
    get_data()
    draw_pale()

if __name__ == '__main__':
    run()