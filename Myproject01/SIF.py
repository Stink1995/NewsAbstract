#!/usr/bin/env python
# coding: utf-8

from sklearn.decomposition import TruncatedSVD
import pkuseg
import numpy as np


seg = pkuseg.pkuseg()
# 获取停用词
stopwords = open('/home/aistudio/work/哈工大停用词表.txt').read().split('\n')


def cutSentence(sentence):
    """
    分词函数：使用北大pkuseg进行分词 同时去除停用词
    """
    tokens = seg.cut(sentence)
    return [t for t in tokens if t not in stopwords]


class SIF(object):

    def __init__(self,model,a=1e-3,unlisted_word_freq=1e-8):
        self.model = model
        self.a = a
        self.unlisted_word_freq = unlisted_word_freq

    def getWordWeight(self):
        """
        统计词库中每个词的频率
        """
        self.frequency = {} 
        vlookup = self.model.wv.vocab
        vocab_count = 0
        for w in vlookup:
            vocab_count += vlookup[w].count

        for k in vlookup:
            self.frequency[k] = vlookup[k].count / vocab_count

    def getWeightAvg(self,sentence):
        """
        获取每句话的加权平均句向量
        """
        token_list = cutSentence(sentence)
    
        vlookup = self.model.wv.vocab
        vectors = self.model.wv
        embedding_size = self.model.vector_size

        vs = np.zeros(embedding_size)
        sentence_length = len(token_list)
        for word in token_list:
            if word in vlookup:
                a_value = self.a / (self.a + self.frequency[word])
                vs = np.add(vs,np.multiply(a_value,vectors[word]))
            else:
                a_value = self.a / (self.a + self.unlisted_word_freq)
                vs = np.add(vs,np.multiply(a_value,np.zeros(embedding_size)))

        vs = np.divide(vs,sentence_length)

        return vs

    def getPCA(self,title,news_split):
        """
        提取标题和新闻正文的一个主成分
        """
        X = []
        vector_t = self.getWeightAvg(title)
        X.append(vector_t)
        for s in news_split:
            vector_s = self.getWeightAvg(s)
            X.append(vector_s)
        mat_X = np.matrix(X)

        svd = TruncatedSVD(n_components=1, n_iter=5, random_state=0)
        svd.fit(mat_X)
        self.pca = svd.components_

        return self.pca

    def getSifVector(self,sentence):
        """
        计算SIF句向量
        """
        vector = self.getWeightAvg(sentence)
        sif_vector = vector - np.dot(self.pca, self.pca.T) * vector
        return sif_vector