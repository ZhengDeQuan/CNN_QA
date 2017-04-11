#!/usr/bin/python
# -*- coding:UTF-8 -*-
'''
cnn网络获得句子向量：
将句子中的每个单词用词向量替换，reshape成一个矩阵，矩阵的行数就是单词数，
矩阵的列数就是词向量的维度，使用不同大小的filter，
无论如何filter的width始终==词向量的维度，filter沿竖直方向滑动，得到每一张图的
卷积的结果是一个（sequence_len - filer_height + 1 , 1）维度的向量，
sequence_len 是句子的长度，filter_height是filter的高度
然后对于这个向量，使用max或者mean pooling
这里的pooling_size，选择的就是(sequence_len - filter_height , 1)
那么这样之后就得到一个（1,1）的scalar
这样就相当于将一个句子用这个filter抽取之后，只是得到了一个数值，这个数值可能描述
的是这个和、句基于当下这个filter的特征。
只是一个数怎么够呢，所以我们有了不止一个filter，而是num_filter个filter，
那么这样之后就会有num_filter个scalar，可以拼接起来作为一个向量，来作为句子的向量。
但是只用一种尺寸的filter得到num_filter个特征可能有点不好
所以可以设置n种不同尺寸的filter得到 n 个num_filter维度的向量，再把这n个向量
首尾拼接起来得到一个长向量，以此作为句子的向量。
***********************************************
得到句子向量后就可以再根据句子向量判断问答句子是否匹配了
余弦相似度，词共现数，weighted词共现数(每个词根据idf值得到一个 weight)，欧拉距离 and so on
这里使用余弦相似度
评价标准和loss是精确度，precision = TP / (TP + FP)；
***********************************************
输入的句子依旧是字符串形式，在本程序中会先将其转换为id表示。
'''
from __future__ import print_function;
import os,sys, timeit,random,operator
import numpy as np
import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
import numpy as np
import numpy
import cPickle
theano.config.floatX = 'float64'
import time
ISOTIMEFORMAT = "%Y - %m - %d %X"

def printf(strr):#传入的参数是要写入文件中的字符串
    f = open("cnn_result.txt","a")#以追加的方式写入，这样不会覆盖掉之前的内容
    print(time.strftime(ISOTIMEFORMAT,time.localtime()) , file = f);#写入时间
    print(strr,file = f);
    f.close();

filenames = [r"WikiQASent-dev",
             r"WikiQASent-dev-filtered",
             r"WikiQASent-test",
             r"WikiQASent-test-filtered",
             r"WikiQASent-train"]

folder = "data_cnn";
trainfile = "-train.txt"
validfile = "-dev.txt"
testfile = "-test.txt"
trainfile = os.path.join(folder,trainfile);
validfile = os.path.join(folder,validfile);
testfile  = os.path.join(folder,testfile);

word_id = cPickle.load(open(os.path.join(folder,"word_id.pkl"),"rb"))
word_vec = cPickle.load(open(os.path.join(folder,"word_vec.pkl"),"rb"));
id_vec = cPickle.load(open(os.path.join(folder,"id_vec.pkl"),"rb"));
def load_data_list(filename):#将语料数据都取出来放在列表中
    List = [];
    with open(filename,"r") as f:
        filelines = f.readlines();
    for line in filelines:
       temp = line.strip("\n").strip().split("\t",5);#que ,ans, label ,qid, wordco;
       List.append([ temp[0],temp[1],int(temp[2]),int(temp[3]) ]);
    return List;


def encode_sent(string,size = 236):
    #句子列表，元素时字符串，改成元素是id，句子的最长的长度是236，但是这个长度是包含了，。？；！等句子中的符号的，符号用<unk>代表
    global word_id;
    x = [];
    for i in range(0,size):
        x.append(word_id["<NULL>"])
    words = string.strip().split();
    for i in range(len(words)):
        if words[i] in word_id:
            x[i] = word_id[words[i]];
        else:
            x[i] = word_id["<UNK>"]
    return x;


def load_batch_data(List, start_pos, batch_size):#将List中的数据分批取出来，每一批有batch_size个，如果最后剩余的少于batch_size个则取[-batch_size:]
    que ,ans ,label = [], [], [];
    total_len = len(List);
    Temp = [];
    if start_pos + batch_size <= total_len :
        Temp = List[start_pos:start_pos + batch_size];
    else :
        Temp = List[ -batch_size :];
    
    for i in range(batch_size):
        q = Temp[i][0];
        a = Temp[i][1];
        l = int(Temp[i][2]);
        
        que.append(encode_sent(q,236));
        ans.append(encode_sent(a,236));
        label.append(l);
       
    return que,ans,label;


    
def clean_word(word):
    word = word.strip(",").strip(".").strip("?").strip(":").strip(";").strip("!").strip("(").strip(")");

def validation(validate_model , test_List , batch_size):
    score_list , index = [] , int(0);
    avg_COST ,tot_COST = float(0) , float(0);
    while True:
        que , ans , label = load_batch_data(test_List ,index , batch_size);
        COST , batch_scores , _ , _ = validate_model(que,ans,label,1.0);#在验证的时候不dropout,所以最后一个是1
        tot_COST += COST;
        for score in batch_scores:#这里的score指的是cos_sim搬移，压缩到0~1之间之后的值
            score_list.append(score);
        index += batch_size;
        if index >= len(test_List):
            break;
        print("validating...",index);
    avg_COST = tot_COST / len(test_List);
    #构造可以用于计算MAP，MRR，precision的sdict
    sdict , index = {} , int(0);
    for items in test_List:#item = [que , ans ,label , qid]
        qid = items[3];
        if not qid in sdict:
            sdict[qid] = [];
        sdict[qid].append((score_list[index] , items[2]))#将“余弦值”和label作为元组加入到sdict[qid],这个列表中
        index += 1;

    #计算precision
    lev0 , lev1 = float(0) ,float(0)
    for qid , cases in sdict.items():
        cases.sort(key = operator.itemgetter(0) , reverse = True)#按照余弦值的大小降序排序
        score , flag = cases[0] #取最高的得分,和对应的label
        if flag == 1:
            lev1 += 1;
        if flag == 0:
            lev0 += 1;
    precision = lev1 /(lev0 + lev1)

    #计算MRR
    MRR , num_Q , reciptrocal_rank = float(0) , float(0) , float(0);
    for qid ,cases in sdict.items():
        #已经拍过序了
        for j in range(len(cases)):
            if cases[j][1] == 1:
                reciptrocal_rank = 1.0 /(j + 1);
                break;
        if reciptrocal_rank != 0.0:
            MRR += reciptrocal_rank ;
            num_Q += 1;
    if num_Q != 0.0:
        MRR /= num_Q;
    else :
        MRR = 0.0;

    #计算MAP
    MAP , num_Q , single_avg , single_num  = float(0) , float(0) , float(0) , float(0);
    #single_avg ：单个主题的平均准确率的平均值
    for qid ,cases in sdict.items():
        #cases.sort(key = operator.itemgetter(0) , reverse = True)
        single_avg = 0.0;
        single_num = 0.0;
        for j in range(len(cases)):
            if cases[j][1] == 1:
                single_num += 1;
                single_avg += single_num/(j + 1);
        if single_num != 0.0 and single_num != len(cases):
            single_avg /= single_num;
        else:
            single_avg = 0.0;

        if single_avg != 0.0:
            MAP += single_avg ;
            num_Q += 1;
    if num_Q != 0.0:
        MAP /= num_Q;
    else :
        MAP = 0.0;
            
    return  tot_COST , avg_COST , MAP , MRR , precision; 
            
    
    
class QACNN(object):
    def __init__(self,que, ans, label, word_embeddings , batch_size,max_sequence_len,embedding_size , filter_sizes , num_filters, keep_prob):
        '''
        input1,inpu2  每一行是一个id化的句子,batch_size行 max_sequence_len 列；
        max_sequence_len:最长的句子的长度
        '''
        rng = np.random.RandomState(23455);
        self.params = [];
        id_vec = theano.shared(name = "id_vec" , value = word_embeddings);
        que_matrix = id_vec[T.cast(que.flatten() , dtype = "int32")]#flatten()之后就是batch_size * max_sequence_len列，一行，但是再从id_vec矩阵中析取之后就是batch_size *max_sequence_len行，embedding_size列了
        ans_matrix = id_vec[T.cast(ans.flatten() , dtype = "int32")]#每max_sequence_len行构成一个句子。

        que_x = que_matrix.reshape((batch_size , 1 , max_sequence_len , embedding_size))
        ans_x = ans_matrix.reshape((batch_size , 1 , max_sequence_len , embedding_size))#为了进行卷积，将原始数据构成这样的格式[batch_size, num_feature_maps,img_height,mig_width]

        self.dbg_x1 = que_x;

        que_vec , ans_vec = [] , [];
        #filter_sizes是一个列表有多种大小的filter_height;
        #对于每一种filter之间并不是级联的关系，而是并列关系，每个filter的width都==embedding_size，然后池化层又都是利用(max_sequence_len - filter_height + 1 , 1)大小的池化块来池化，所以最后每个filter的情况下得到的都是num_filters维的向量
        for i,filter_size in enumerate(filter_sizes):
            filter_shape = (num_filters , 1 , filter_size , embedding_size)#根据函数conv2d(A,W)的要求，W[目标特征数，源特征数，filter_height, filter_width]
            img_shape =(batch_size , 1 , max_sequence_len , embedding_size)#A[batch_size , 图片特征数，img_height , img_width]
            fan_in = np.prod(filter_shape[1:])#每一层的神经元接收的上一层的输入的数量是[上一层的feature_map数 * filter_height * filter_width]
            fan_out = np.prod(filter_shape[2:]) * filter_shape[0] // (max_sequence_len - filter_size + 1)
            W_bound = np.sqrt(6./(fan_in + fan_out))
            W = theano.shared(
                name = "W_%d"%i,
                value =
                    np.array(
                    rng.uniform(low = - W_bound , high = W_bound , size = filter_shape),
                    dtype = theano.config.floatX
                    ),
                borrow = True)
            b_values = np.zeros((filter_shape[0],),dtype = theano.config.floatX)
            b = theano.shared(name = "b_%d"%i , value = b_values,borrow = True)

            #卷积+max_pooling
            conv_out = conv2d(input = que_x , filters = W , filter_shape = filter_shape , input_shape = img_shape)
            #卷积后[batch_size , num_filters , max_sequence_len - filer_size + 1 , 1]
            pooled_out = pool.pool_2d(input = conv_out , ds = (max_sequence_len - filter_size + 1 , 1) , ignore_border = True , mode = "max")
            #池化后[batch_size , numfilter , 1 , 1]
            pooled_active = T.tanh(pooled_out + b.dimshuffle('x' , 0 , 'x' , 'x'))
            que_vec.append(pooled_active)


            conv_out = conv2d(input = ans_x , filters = W , filter_shape = filter_shape , input_shape = img_shape)
            pooled_out = pool.pool_2d(input = conv_out , ds = (max_sequence_len - filter_size + 1 , 1) , ignore_border = True , mode = "max")
            pooled_active = T.tanh(pooled_out + b.dimshuffle('x' , 0 , 'x' , 'x'))
            ans_vec.append(pooled_active)
            self.params += [W,b]
            self.dbg_conv_out = conv_out.shape

        num_filters_total = num_filters * len(filter_sizes)
        self.dbg_outputs_que = que_vec[0].shape
        que_vec_flat = T.reshape(T.concatenate(que_vec , axis = 1) , [batch_size , num_filters_total])
        ans_vec_flat = T.reshape(T.concatenate(ans_vec , axis = 1) , [batch_size , num_filters_total])
        #问答句向量到此计算结束

        drop_que = self._dropout(rng , que_vec_flat , keep_prob)
        drop_ans = self._dropout(rng , ans_vec_flat , keep_prob)
        #drop_out

        #计算问题与答案之间的余弦相似度
        len_que = T.sqrt(T.sum(drop_que * drop_que , axis = 1))
        len_ans = T.sqrt(T.sum(drop_ans * drop_ans , axis = 1))
        cos_sim = T.sum(drop_que * drop_ans , axis = 1) / (len_que * len_ans)
        
        self.cos_sim = cos_sim;
        #损失
        def loss(label , cos_sim , batch_size):
            cos_sim = (cos_sim + 1.0)/2.0 #搬移，压缩到（0~1）之间,
            cos_sim = T.clip(cos_sim , 1e-10 , 1-1e-10)#对0有效，对1无效
            cross_entropy_loss = T.nnet.binary_crossentropy(cos_sim, label)
            ones = theano.shared(numpy.ones(batch_size , dtype = theano.config.floatX), borrow = True);
            cross_entropy_loss = T.switch(T.eq(cos_sim, ones),- ( label * T.log(cos_sim) + (1 - label) * T.log(1e-10) ),cross_entropy_loss)
            return cross_entropy_loss;

        
        cross_entropy_loss = loss(label , cos_sim , batch_size);
        self.cost = T.sum(cross_entropy_loss , acc_dtype = theano.config.floatX);
        
        
    def _dropout(self , rng , layer , keep_prob):
        srng = T.shared_randomstreams.RandomStreams(rng.randint(123456));
        mask = srng.binomial(n = 1, p = keep_prob , size = layer.shape)#二项分布，成功地概率是p,实验次数n，返回成功地次数
        output = layer * T.cast(mask,theano.config.floatX)
        output = output / keep_prob;
        return output;
    

    def save(self,folder = "save_my_params_cnn"):
        if not os.path.exists(folder):
            os.mkdir(folder);
        for param in self.params:
            numpy.save(os.path.join(folder,param.name + ".npy"),param.get_value());


    def load(self,folder = "save_my_params_cnn"):
        for param in self.params:
            param.set_value(numpy.load(open(os.path.join(folder , param.name + ".npy"))));

    def Show(self):
        for param in self.params:
            print(param.name , param.get_value());

            
if __name__=="__main__":
    params = {
        "batch_size"  : int(256),#一次训练的句子数
        "filter_size" : [2,3,4],
        "num_filters" : 500,
        "embedding_size" : 300,#词向量的维度
        "learning_rate" : 0.0005,#原来是0.001，感觉效果不好
        "n_epoches" : 1000,
        "valid_frequency" : 79,#在下面训练前会改为n_epoch
        "keep_prob" : 0.5,#used for drop out
        "max_sequence_len" : 236
        };
    printf("params = " + str(params));
    #将function的定义放在类外，将计算图的定义放在类内，这样我觉得更好
    
    que  , ans , label = T.matrix("que") , T.matrix("ans") , T.vector("label");
    keep_prob = T.fscalar("keep_prob");#直接参与运算的都需要是tensor
    model = QACNN(que = que , ans = ans , label = label , keep_prob = keep_prob ,word_embeddings = id_vec ,
                  batch_size = params['batch_size'] , max_sequence_len = params['max_sequence_len'] ,
                  embedding_size = params['embedding_size'] ,filter_sizes = params["filter_size"] , num_filters = params["num_filters"])

    dbg_x1 = model.dbg_x1;# = que_x
    dbg_outputs_que = model.dbg_outputs_que # = que_vec[0].shape
    #在类中只是将计算图定义完了，计算图的真正的启动--定义函数输入输出以触发图的计算，还有梯度反传的定部分还没有定义，
    #梯度反传和计算图之间是要通过function的定义联系到一起的。
    
    cost , cos_sim = model.cost , model.cos_sim;
    graph_params = model.params;
    grads = T.grad(cost , graph_params);
    learning_rate = T.dscalar("learning_rate");
    updates = [(param_i , param_i - learning_rate * grad_i) for param_i , grad_i in zip(graph_params ,grads) ]

    qt , at , lt = T.matrix("q1") , T.matrix("a1") , T.vector("l1");
    prob = T.fscalar("prob");
    train_model = theano.function(inputs = [ qt , at , lt , prob , learning_rate] ,
                                  outputs = [ cost , dbg_x1 , dbg_outputs_que],
                                  updates = updates,
                                  givens = {
                                      que : qt , ans : at ,label : lt , keep_prob : prob
                                      })
    
    qv, av , lv = T.matrix("qv") , T.matrix("av") , T.vector("lv");
    validate_model = theano.function(inputs = [qv , av , lv , prob],
                                     outputs = [cost , cos_sim , dbg_x1 , dbg_outputs_que],
                                     #updates = updates,
                                     givens = {
                                         que : qv , ans : av , label : lv , keep_prob : prob
                                         })
    train_List = load_data_list(trainfile);
    valid_List = load_data_list(validfile);
    printf("train_start");
    print("train_start")
    
    n_train_batches = len(train_List) // params["batch_size"];
    params["valid_frequency"] = n_train_batches + 1;# =78 + 1
    params["patience"] = int((n_train_batches + 1)*1.5);
    epoch = 0;
    done_looping = False;
    last_pos = 0;
    best_MAP = -numpy.inf;
    best_MRR = -numpy.inf;
    best_COST = numpy.inf;
    best_PREC = -numpy.inf;
    
    index = 0;
    valid_times = 0;
    last_update_valid_time = 0;
    while(epoch < params["n_epoches"] )and (not done_looping):
        epoch = epoch + 1;
        print("epoch = ",epoch);
        for minibatch_index in range(n_train_batches + 1):
            index += 1;
            print("index = ",index);
            train_que,train_ans,train_label = load_batch_data(train_List,last_pos,params['batch_size']);
            last_pos = last_pos + params["batch_size"];
            if last_pos >= len(train_List):
                last_pos = last_pos % len(train_List);
            cost , dbg_x1 , dbg_outputs_que = train_model(train_que, train_ans , train_label , params["keep_prob"],params['learning_rate'])
            printf("epoch : " + str(epoch) + " index :" + str(index) + " cost = " + str(cost));
            print("epoch : " + str(epoch) + "index :" + str(index) + " cost = " + str(cost))
            if index % params["valid_frequency"] == 0:
                valid_times += 1;
                flag = False
                printf("validation....")
                print("validation....")
                COST , avg_COST , MAP , MRR , PREC = validation(validate_model , valid_List , params["batch_size"])
                printf("tot_COST=" + str(COST) + " avg_COST=" + str(avg_COST) + " MAP=" +str(MAP) + " MRR=" +str(MRR) + "prec="+str(PREC));
                print("tot_COST=" + str(COST) + " avg_COST=" + str(avg_COST) + " MAP=" +str(MAP) + " MRR=" +str(MRR) + "prec="+str(PREC));
                if MAP > best_MAP:
                    best_MAP = MAP;
                    flag = True;
                if MRR > best_MRR:
                    best_MRR = MRR;
                    flag = True;
                if avg_COST < best_COST:#因为最后一个batch可能句子数不足
                    best_COST = avg_COST;
                    flag = True;
                if PREC > best_PREC:
                    best_PREC = PREC;
                    flag = True;
                if flag:
                    last_update_valid_time += 1;
                    printf("last_update_valid_time = " + str(last_update_valid_time) + " best_cost = "+ str(best_COST));
                
                if ( valid_times - last_update_valid_time ) > params["patience"] :#如果过了很久都没有改进就没有必要再训练下去了,我这里设置的是，trian中全部batch都训练完了，又在下一个epoch训练了半个训练集还没有更新的话，就将学习率递减/2
                    params['learning_rate'] /= 2.0;
                    last_update_valid_time = valid_times;#lr/2后要将两者的差值变为0；
                if (params['learning_rate'] < 1e-8 ):
                    done_looping = True;
                    break;
            
    printf("train_over");
    print("train_over");
    model.save();
    ##########
    #测试模型#
    ##########
    del train_List;
    del valid_List;
    test_List = load_data_list(testfile)
    test_List
    COST , avg_COST , MAP , MRR , PREC = validation(validate_model , test_List ,params["batch_size"]);
    printf("test....");
    print("test....");
    printf("tot_COST=" + str(COST) + " avg_COST=" + str(avg_COST) + " MAP=" +str(MAP) + " MRR=" +str(MRR) + "prec="+str(PREC));
    print("tot_COST=" + str(COST) + " avg_COST=" + str(avg_COST) + " MAP=" +str(MAP) + " MRR=" +str(MRR) + "prec="+str(PREC));
    printf("test over")
    print("test over")
