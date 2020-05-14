#!/usr/bin/env python
# coding: utf-8

from gensim.models import Word2Vec
import re
import pkuseg
import numpy as np
from get_keywords import *
from SIF import *
from collections import Counter
from sklearn.decomposition import TruncatedSVD



def cutNews(para):
    """
    对新闻正文进行分句
    """
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)               # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)                   # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)                   # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)    
    # 如果双引号前有终止符，那么双引号才是句子的终点，
    #把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()                                               # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，
    #需要的再做些简单调整即可。
    return para.split("\n")

def cutSentence(sentence):
    """
    分词函数：使用北大pkuseg进行分词 同时去除停用词
    """
    tokens = seg.cut(sentence)
    return [t for t in tokens if t not in stopwords]


def keywordCountInSentence(sentence_list,keywords):
    """
    计算每一句话中包含的关键词的数量
    """
    outputs = {}
    for sentence in sentence_list:
        tokens = cutSentence(sentence)
        token_count = Counter(tokens)
        keywords_count = 0
        for keyword in keywords:
            if keyword in token_count:
                keywords_count += token_count[keyword]
            else:
                keywords_count += 0
        outputs[sentence] = keywords_count
    return outputs


def allKeywordsCount(news,keywords):
    """
    计算出每个关键词在news中的总数
    """
    news_tokens = cutSentence(news)
    count = Counter(news_tokens)
    keywords_count = 0
    for keyword in keywords:
        keywords_count += count[keyword]
    return keywords_count


def sentenceWeight(news,news_split,keywords):
    """
    根据每句话中所包含的关键词的数量计算权重
    """
    count = keywordCountInSentence(news_split,keywords)
    all_count = allKeywordsCount(news,keywords)
    output = {}
    for k,v in count.items():
        weight = v / all_count
        output[k] = weight
    return output


def cosinSimilarity(X1,X2):
    """
    计算余弦相似度
    """
    num = float(np.dot(X1,X2.T))
    denom = np.linalg.norm(X1) * np.linalg.norm(X2)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    
    return sim


def getSimilarity(Vt,Vc,Vs):
    """
    计算新闻短句与新闻标题和新闻正文长句的相似度
    与标题相似性权重:0.2
    与新闻正文长句相似性权重:0.8
    权重系数可以自己精心调参
    """
    Ct = compute_similarity(Vt,Vs)
    Cn = compute_similarity(Vc,Vs)
    sim = (0.2 * Ct + 0.8 * Cn) / 2
    return sim


# 句子的平滑  取每句话在新闻中位置前后两句话的相似度与自身相似度之和加权平均
def knnSmooth(scores,keywords_weight):
    """
    输入scores是[(短句，score)....(短句,score)]的形式
    句子平滑 取每句话在新闻中位置前后两句话的相似度与自身相似度之和加权平均
    """
    scores_list = []
    for t in scores:
        scores_list.append(t[1])
    
    output = []
    for i in range(len(scores)):
        if i == 0:       
            # 句首一般为开篇的话语，应该赋予相对较高一点的权重，乘以1.1的系数
            score_avg = ((scores_list[0] + scores_list[1]) / 2) * 1.1
        elif i == (len(scores) - 1):   
            # 句尾一般也为总结性的话语，也相应的赋予较高的权重，乘以 1.1的系数
            score_avg = ((scores_list[-2] + scores_list[-1]) / 2 ) * 1.1
        else:
            score_avg = (scores_list[i-1] + scores_list[i] + scores_list[i+1]) / 3

        smooth_score = keywords_weight[scores[i][0]]*score_avg
        
        output.append((scores[i][0],smooth_score))

    return output

def sortSentence(sc):
    """
    按照得分进行排序
    """
    nv = sorted(sc,key=lambda x:x[1],reverse = True)
    return nv


def getAbstract(sort_c, news,sentence_list):
    """
    按照新闻长度的30%取出作为我们的摘要
    """
    length = len(sort_c)
    # 太短的news直接将全篇新闻当做摘要输出
    if length <= 5:
        return news
    else:
        abstract_list = []
        # 取出新闻内容的30%作为摘要输出
        top_n = length // 3
        for i in range(top_n):
            abstract_list.append(sort_c[i][0])
        output = []
        for s in sentence_list:
            if s in abstract_list:
                output.append(s)
        abstract = "".join(output)
        return abstract





if __name__ == "__main__":

	news = '如今的中国互联网，可谓是精彩纷呈，每天都有新的奇迹发生，每天都有精彩的故事演绎。滴滴，美团，字节跳动，拼多多，这些互联网新秀们如同少年天才一般，用自己的天赋异禀，一次次让世人惊叹。但是不管这些新秀们多么出彩，BAT却是他们一直绕不开的“槛”。作为中国最早一批的互联网公司，BAT早就脱离了互联网范畴，进而渗透到我们生活的方方面面。BAT，即百度，阿里巴巴，腾讯三家互联网公司。三家公司把持着搜索，支付，社交等互联网领域，这三钟领域，也被称为互联网的“三重门”。意味着如果要上网，必定要经过这三扇门。这也很好理解，我们平时上网，无非就是找资料，玩游戏，购物，沟通交流，玩游戏等。经过20年的发展，三家互联网公司也在自己的领域形领先优势，并且逐渐扩展到其他领域，例如腾讯，利用社交优势，在游戏方面做到行业前列，阿里巴巴利用支付和购物这一块的优势，在物流和办公领域走到前列。不过百度最近却“身陷囹圄”，其实百度曾经是三者之间最有机会做大做强的。如果说搜索，支付，社交是互联网“三重门”，那么搜索必定是第一扇门。早期的网友根本不是很了解阿里巴巴和腾讯，所以最早上网主要是用百度查资料。只是百度没有利用好这么多的用户基础，而在自己拳头产品——百度搜索里面植入太多的不良广告，并且在竞价排名上太过注重利益，导致用户口碑一路下滑。“魏则西事件”则是压垮百度口碑的最后一根稻草，虽然百度很快将负责广告业务的副总裁辞退，并且也优化了广告展示这一块。但是信任就像“信用卡”，一味的透支用户的信任，用户最后也就不会相信你了，这也是现在百度一蹶不振的主要原因。反观阿里巴巴和腾讯，自己的拳头产品—淘宝和QQ，虽然也一直存在商业化，但是在产品上，确确实实的为广大用户提供了便利，改变了人们的生活，并且在用户口碑上，也没出现过大的崩塌（当然小学生充Q币和淘宝假货也算是这两家公司的黑点）。百度的一蹶不振，也让一直被互联网三座大山压的喘不过气的新秀们看到了一丝曙光。阿里和腾讯，一家直接主宰了人们的互联网消费，一家直接主宰了人们的互联网通讯，之前也有过互联网公司像两者发起挑战，但是要么失败，如米聊，要么就是被这两家公司打上他们的标签，如京东。而百度的搜索领域，由于搜索本身的局限性，导致百度一直没在其他领域有效的扩张，加上之前的口碑崩塌，在市值方面，百度早已经和阿里腾讯不是“同一个世界”了。而互联网新秀们抓住机会，再一次向三座大山发起了进攻，这次的进攻对象也非常明确，就是百度。而挑战者，则是美团和蚂蚁金服，这也就是前面说的，ATM里面的M。两个M里面，我们更加熟悉美团，毕竟现在是“外卖时代”，人们或多或少都点过美团的外卖。但是对于美团来说，外卖从来就不是美团的主业，而美团的主业就是什么都做。从最开始的外卖，到后面的团购拼单，人们本来以为美团会做一家本地化生活服务，但是美团却出其不意的做起酒店预订，网约车，互联网贷款等业务。对于美团来说，通过前期外卖业务打造的口碑，美团有了一定的用户基础。但美团的野心却不止于此，他也想效仿阿里和腾讯，做到全覆盖。只是想法虽好，但是步子迈的稍微有点大。“老江湖”的阿里和腾讯难道不知道美团的想法？于是一直扶持美团的阿里，慢慢取消了扶持，而喜欢投资的腾讯，现在也逐渐减少了对美团的投资。但是对于美团来说，目前气候已成，如何“上位”，就要看持续的表现了。另一个M，蚂蚁金服，其实就是支付宝的母公司，我们又爱又恨的“花呗”“借呗”也正是来自于这家公司。看到这里，人们不禁想问，那蚂蚁金服不就是阿里的子公司了？其实并不然，虽然阿里巴巴持有蚂蚁金服的股份，但是蚂蚁金服早在2014年就独立了出来，自负盈亏。也就是说，阿里从此专注于购物消费平台，而蚂蚁金服则专注于金融支付平台，并且和国家很多部门以及银行都有着深度的合作。蚂蚁金服也被认为是最有可能“上位”的，在2018年互联网公司市值排名中，蚂蚁金服就达到了1510亿美元，虽然远远不及腾讯和阿里的5000亿美元，但是也领先于百度的920亿美元。但是笔者认为，BAT并不是单纯的“以市值论英雄”，更多时候已经成为人们的一种习惯。市值这东西，如果腾讯独立出一家公司，凭借腾讯的资源，也会很快达到几百甚至上千亿元的市值，但是在很长一段时间内，用户还是习惯用百度，腾讯和阿里的产品。百度似乎也知道自己掉队已久，所以这几年也发力AI人工智能，随着5G时代的到来，5G+AI必定会成为互联网一个新的增长点。到那时，酝酿已久的百度，能否凭借着5G+AI，追上阿里和腾讯了？'
	title = '以前的BAT，现在变成了ATM，是尘埃落定还是为时尚早？'
	seg = pkuseg.pkuseg()
	# 获取停用词
	stopwords = open('./data/stopwords/哈工大停用词表.txt').read().split('\n')

	# 加载word2vec模型
	model = Word2Vec.load('/home/aistudio/work/wiki_word2vec.model')
	# 对新闻进行分句
	news_split = cutNews(news)

	# 提取关键词
	tr = TextRank(news,0.85, 700)
	tr.cutSentence()
	tr.createNodes()
	tr.createMatrix()
	tr.calPR()
	tr.Result()
	keywords = tr.getKeyWords()
	# 计算每句话包含关键词的权重
	sentence_weight = sentenceWeight(news,news_split,keywords)
	# 获取SIF句向量
	sif = SIF(model)
	sif.getWordWeight()
	sif.getWeightAvg(title)
	sif.getPCA(title,news_split)
	
	vt = sif.getSifVector(title)
	vc = sif.getSifVector(news)
	scores = []
	for s in news_split:
	    vs = sif.getSifVector(s)
	    score = getSimilarity(vt,vc,vs)
	    scores.append((s,score))
	# 进行平滑之后的得分
	smooth_scores = knnSmooth(scores,sentence_weight)

	# 对得分进行排序
	sort_sentence = sortSentence(smooth_scores)

	# 获取摘要
	getAbstract(sort_sentence,news,news_split)

