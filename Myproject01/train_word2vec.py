#!/usr/bin/env python
# coding: utf-8

"""
使用gensim训练Word2Vec词向量
"""
from gensim.models import word2vec

if __name__ == "__main__":
	"""
	参数说明：
	size：词向量维度
	window：滑动窗口宽度
	min_count：最小词频 词频低于5就过滤掉
	iter：迭代多少次
	workers：进程数
	sg：采用CBOW还是Skip-gram 0：CBOW 1：skip-gram
	"""
	sentences = word2vec.LineSentence("./data/corpus_split.txt")
	model = word2vec.Word2Vec(sentences,size=300,window=5,min_count=5,iter=7,workers=8,sg=0)
	# 保存模型
	model.save("./wiki_word2vec.model")