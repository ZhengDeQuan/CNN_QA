#!/usr/bin/python
# -*- coding:UTF-8 -*-
'''
最简单原则，将全部的google有对应的向量的单词装到word_vec中，
训练语料中出现的，但是googl的word2vec模型中没有的，直接用<unk>代表向量的值就是0.01

m = 236 也就是说，最长的句子有236个单词，但是这个236是包含所有的符号的，比如句子中的逗号，句末的句号或者感叹、问号等。
'''
import os
import gensim
import cPickle
import numpy

word_vec = {};
word_id = {};
id_vec = [];
nonword_id = {}#有些单词在model没有，可是在train语料中有，这些单词可能就是我们需要进行训练的。但是考虑到所占比重比较少，也可以不算。
nonid_vec =[];

filenames = [r"WikiQASent-dev",
             r"WikiQASent-dev-filtered",
             r"WikiQASent-test",
             r"WikiQASent-test-filtered",
             r"WikiQASent-train"]

model = gensim.models.Word2Vec.load_word2vec_format('../../Code_and_Corpus/GoogleNews-vectors-negative300.bin',binary = True);

def make_word_vec():
    global filenames;
    global word_vec;
    m = 0;
    for filename in filenames:
        filename = filename + ".txt";
        with open(filename,"r") as f:
            filelines = f.readlines();
        for line in filelines:
            temp = line.rstrip('\n').split('\t',3);#que , ans , label;
            question = temp[0].strip().split();
            if len(question) > m:
                m = len(question)
            answer = temp[1].strip().split();
            if len(answer) > m :
                m = len(answer);
            for word in question :
                word.strip(",").strip(".").strip("?").strip("!").strip(":")
                if word in model:
                    if word not in word_vec:
                       word_vec[word] = model[word];
                '''
                else :
                    words = word.split('-');#可能有些比如warm-up是由连字符连接的
                    for w in words:
                        if w in model:
                            if w not in word_vec:
                                word_vec[w] = model[w];
                '''
                    
            for word in answer:
                word.strip(",").strip(".").strip("?").strip("!").strip(":")
                if word in model:
                    if word not in word_vec:
                        word_vec[word] = model[word];
                '''
                else :
                    words = word.split('-');
                    for w in words:
                        if w in model:
                            if w not in word_vec:
                                word_vec[w] = model[w];
                '''
                    
    print("number of words in word_vec = ",len(word_vec));#26942
    cPickle.dump(word_vec,open('word_vec.pkl',"wb"));
    print("over");
    print("m = ",m);

def make_word_id():
    global filenames;
    global word_id;
    global word_vec;
    myid = 0;
    word_id["<NULL>"] = myid;
    myid += 1;
    word_id['<UNK>'] = myid;
    myid += 1;
    for word in word_vec:
        if word not in word_id:
            word_id[word] = myid;
            myid += 1;
    print("number of words in word_id = ",len(word_id));#41985
    cPickle.dump(word_id,open("word_id.pkl","wb"));
    print("over");


def make_id_vec():
    global word_vec;
    global word_id;
    global id_vec;

    tempvec = [];
    for i in range(300):
        tempvec.append(0.01);
    for i in range(len(word_id)):
        id_vec.append(tempvec);
    
    tempvec = [];
    for i in range(300):
        tempvec.append(0);
    id_vec[0] = tempvec;
    
    tempvec = [];
    for i in range(300):
        tempvec.append(0.01);
    id_vec[1] = tempvec;

    
    
    for word,myid in word_id.items():
        if word in word_vec:
            id_vec[myid] = word_vec[word];
        else :
            id_vec[myid] = tempvec;
            
    #将字典转化为矩阵，id是行下标，每一行是一个词向量。
    id_vec = numpy.array(id_vec);
    print("number of vec in id_vec ", id_vec.shape);
    cPickle.dump(id_vec,open("id_vec.pkl","wb"));
    print("over");

    
if __name__ =="__main__":
    
    make_word_vec();
    make_word_id();
    make_id_vec();
    #把在word_id中但是不在word_vec中的单词，挑出来，构造nonword_id,在构造nonid_vec,nonid_vec中的向量需要在每次的update是更新，就是要跟着一起被训练
    '''
    word_id = cPickle.load(open("word_id.pkl","rb"))
    word_vec = cPickle.load(open("word_vec.pkl","rb"))
    temp =[];
    for word in word_id:
        if word not in word_vec:
            temp.append(word);
    '''
