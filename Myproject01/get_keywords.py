#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pkuseg

seg = pkuseg.pkuseg()
# 获取停用词
stopwords = open('./data/stopwords/哈工大停用词表.txt').read().split('\n')  

def cutSentence(sentence):
    """
    分词函数：使用北大pkuseg进行分词 同时去除停用词
    """
    tokens = seg.cut(sentence)
    return [t for t in tokens if t not in stopwords]


class TextRank(object):
    
    def __init__(self, sentence, alpha, iternum):
        self.sentence = sentence                  # 输入的待提取关键词的文本
        self.alpha = alpha                        # 平滑系数
        self.edge_dict = {}                       # 记录节点的边连接字典
        self.iternum = iternum                    # 迭代次数
 
    #对句子进行分词
    def cutSentence(self):
        self.word_list = cutSentence(self.sentence) 
 
    #根据窗口，构建每个节点的相邻节点,返回边的集合
    def createNodes(self):
        for index in range(len(self.word_list)-1):
            word = self.word_list[index]
            next_word = self.word_list[index+1]
            tmp_list = []
            if word not in self.edge_dict.keys():
                tmp_list.append(next_word)
                self.edge_dict[word] = tmp_list
            elif next_word not in self.edge_dict[word]:
                self.edge_dict[word].append(next_word)
            else:
                continue
                
 
    #根据边的相连关系，构建邻接矩阵
    def createMatrix(self):
        self.matrix = np.zeros([len(set(self.word_list)), len(set(self.word_list))])
        self.word_index = {}                   #记录词的index
        self.index_dict = {}                   #记录节点index对应的词
 
        for i, v in enumerate(set(self.word_list)):
            self.word_index[v] = i
            self.index_dict[i] = v
        for key in self.edge_dict.keys():
            for w in self.edge_dict[key]:
                self.matrix[self.word_index[key]][self.word_index[w]] = 1
                self.matrix[self.word_index[w]][self.word_index[key]] = 1
        #归一化
        for j in range(self.matrix.shape[1]):
            sum = 0
            for i in range(self.matrix.shape[0]):
                sum += self.matrix[i][j]
            for i in range(self.matrix.shape[0]):
                self.matrix[i][j] /= sum
 
    #根据textrank公式计算权重
    def calPR(self):
        self.PR = np.ones([len(set(self.word_list)), 1])
        for i in range(self.iternum):
            self.PR = (1 - self.alpha) + self.alpha * np.dot(self.matrix, self.PR)
 
    #输出词和相应的权重
    def Result(self):
        word_pr = {}
        for i in range(len(self.PR)):
            word_pr[self.index_dict[i]] = self.PR[i][0]
        self.res = sorted(word_pr.items(), key = lambda x : x[1], reverse=True)

    # 提取输入文本中 5%的词为关键词
    def getKeyWords(self):
        keywords_count = len(self.res) // 20
        keywords = []
        for i in range(keywords_count):
            keywords.append(self.res[i][0])
        return keywords