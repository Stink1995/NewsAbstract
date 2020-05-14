#!/usr/bin/env python
# coding: utf-8
import os
import json
import pkuseg

"""
1.下载wiki中文百科语料
地址：https://github.com/brightmart/nlp_chinese_corpus 
本项目使用语料:维基百科json版(wiki2019zh)
"""
def mergeFile(filepath):
    """
    将多个小文件合并成一个大文件
    """
    save_path = os.path.join(filepath,'corpus.txt')
    save_file = open(save_path,'w+')
    sub_file = ['AA','AB','AC','AD','AE','AF','AG','AH','AI','AJ','AK','AL','AM']
    for f in sub_file:
        path = os.path.join(filepath,f)
        for i in range(100):
            txt_path = os.path.join(path,"wiki_{}".format(str(i).zfill(2)))
            try:
                file = open(txt_path,'r',encoding='utf-8')
                line = file.readline()
                while line:
                    pretreatment_line = json.loads(line)['text'].replace('\n','')
                    save_file.write(pretreatment_line+'\n')
                    line = file.readline()
                file.close()
            except FileNotFoundError:
                print("End")
    save_file.close()

# mergeFile('./data/wiki_zh')

if __name__ == "__main__":
	"""
    使用pkuseg开启8进程对文件进行分词,根据自己的CPU数来设定
    训练Word2Vec词向量时不需要进行去停用词
    所以在切分后也未进行去停用词处理
    """
    pkuseg.test('./data/wiki_zh/corpus.txt', './data/corpus_split.txt', nthread=8)
