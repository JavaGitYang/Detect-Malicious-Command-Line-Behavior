#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
tools_for_text_analysis.py
用于文本分析的工具包
'''

__author__ = "Su Yumo <suym@buaa.edu.cn>"

import random
import re
import operator
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from keras.layers import Input, Dense
from keras.layers import Lambda as keras_Lambda
from keras.models import Model
from keras import regularizers
from keras import backend as K
from keras import metrics as keras_metrics


def Parse_data(dir_of_inputdata,options='one'):
    '''
    对输入的文本数据进行解析
    dir_of_inputdata：输入的文本数据
    options：选择文本解析模式，'one'代表普通模式，'two'代表只选命令行中的关键词
    '''
    dataset = []
    remove_data = ['ls','cd','pwd','cat']
    
    if options == 'one':
        #匹配非字母字符，即匹配特殊字符
        regEx = re.compile('\\W*')
        with open(dir_of_inputdata) as f:
            for line in f.readlines():
                #去掉行尾换行符
                line=line.rstrip('\n')
                listoftoken = regEx.split(line)
                #去掉空格值，且将字符串转变为小写
                tem = [tok.lower() for tok in listoftoken if len(tok)>0]
                #去掉第一个无用的序列号
                del tem[0]
                #去掉字符小于1的值
                tem2 = [b for b in tem if len(b)>1]
                #去掉无用的命令
                tem_data = [a for a in tem2 if a in remove_data]
                if len(tem_data) == 0:
                    dataset.append(tem2)
    if options == 'two':
        regEx = re.compile('\\W*')
        with open(dir_of_inputdata) as f:
            for line in f.readlines():
                #去掉行尾换行符
                line=line.rstrip('\n')
                #按照空格划分字符串
                listoftoken = re.split(' ',line)
                data_tem = []
                for token in listoftoken:
                    #按照/划分一条命令行的字符串
                    tem = re.split(r'/',token)
                    #len(tok)>0是为了能取出/bin/read/ 中的read而不是最后一个/后的空格
                    tem2 = [tok for tok in tem if len(tok)>0]
                    if len(tem2) != 0:
                        #取出一个命令行中关键的命令字段
                        tem3 = tem2[-1]
                    else :
                        #如果token只有/或者空格，例如/ 或者//，那么tem、tem2为空
                        continue
                    #tem3是字符串，不是list，所以用append，而不是extend
                    data_tem.append(tem3)
                #将data_tem中的关键命令字段按照空格相连
                data_tem1 = ' '.join(data_tem)
                
                data_tem2 = regEx.split(data_tem1)
                #去掉空格的值，且将字符串转变为小写,注意不要写len(tok)>1,会对del data_tem[0]有影响，因为1到9序列号字符为1
                data_tem3 = [tok.lower() for tok in data_tem2 if len(tok)>0]
                #去掉第一个无用的序列号
                del data_tem3[0]
                #去掉字符小于1的值
                data_tem4 = [tok for tok in data_tem3 if len(tok)>1]
                #去掉无用的命令
                tem_data = [a for a in data_tem4 if a in remove_data]
                if len(tem_data) == 0:
                    dataset.append(data_tem4)
                
    return dataset

def Merge_data_split_window(dataset):
    '''
    将单条命令行按照时间窗口合并成多条，减小统计误差
    '''
    new_dataset = []
    nums = len(dataset)
    set_num = 10
    count = 0
    tem = []
    for data in dataset:
        tem.extend(data)
        count = count +1
        if count % set_num == 0:
            new_dataset.append(tem)
            tem = []
    #将最后没整除set_num的数据归为一条
    new_dataset.append(tem)
    print "Number of the dataset: %s" % nums
    print "Number of the new_dataset: %s" % len(new_dataset)
    print "Number of the set_num: %s" % set_num
    
    return new_dataset

def Merge_data_sliding_window(dataset):
    '''
    将单条命令行按照时间滑动窗口合并成多条，减小统计误差
    '''
    new_dataset = []
    nums = len(dataset)
    set_num = 10
    count = 0
    new_count = 0
    tem = []
    while count != nums:#要检查一下是否是nums-1
        tem.extend(dataset[count])
        count = count +1
        new_count = new_count +1
        if new_count % set_num == 0:
            new_dataset.append(tem)
            tem = []
            count = count - set_num +1
            new_count = 0
    print "Number of the dataset: %s" % nums
    print "Number of the new_dataset: %s" % len(new_dataset)
    print "Number of the set_num: %s" % set_num
    
    return new_dataset

def Store_data(dir_of_inputdata,dataset):
    '''
    将数据样本写入文本
    dir_of_inputdata：存储文本的文件
    dataset：输入的文本数据
    '''
    with open(dir_of_inputdata,'w+') as f:
        for dataline in dataset:
            f.write(str(dataline)+'\n')

def Mark_data(dataset):
    '''
    将样本数据，按照是否存在'sudo'，分为两类并标记
    dataset：样本数据集
    '''
    classset = [0]*len(dataset)
    i = 0
    for document in dataset:
        if 'sudo' in document: #有可能会出现sudo在'sudo****'，被认为存在sudo的问题
            classset[i] = 1
        i = i + 1
        
    return classset

def CreateVocabList(dataset):
    '''
    利用所有的样本数据，生成对应的词汇库
    dataset：样本数据集
    '''
    vocabset = set([])
    for document in dataset:
        vocabset = vocabset | set(document)
    #去掉sudo这个特殊的字符串
    #vocabset.remove('sudo')
    print 'The length of the vocabulary: %s' %len(vocabset)
    
    return list(vocabset)

def SetofWords2Vec(vocablist,inputset):
    '''
    利用词汇库，将文本数据样本，转化为对应的词条向量
    vocablist：词汇表
    inputset：文本数据集
    '''
    datavec = []
    for document in inputset:
        tem = [0]*len(vocablist)
        for word in document:
            if word in vocablist:
                tem[vocablist.index(word)] = 1
            else:
                print "the word : %s is not in my vocabulary!" % word
        datavec.append(tem)
        
    return datavec

def BagofWords2Vec(vocablist,inputset):
    '''
    利用词汇库，将文本数据样本，按照词频转化为对应的词频向量
    vocablist：词汇表
    inputset：文本数据集
    '''
    datavec = []
    for document in inputset:
        tem = [0]*len(vocablist)
        for word in document:
            if word in vocablist:
                tem[vocablist.index(word)] += 1
            else:
                print "the word : %s is not in my vocabulary!" % word
        datavec.append(tem)
        
    return datavec

def Load_data(datavec,classset):
    '''
    将数据样本划分为训练集和测试集
    参数stratify=Y，分层抽样，保证测试数据的无偏性
    '''
    X = datavec
    Y = classset
    print "Number of the positive class: %s" % classset.count(1)
    print "Number of the negative class: %s" % classset.count(0)
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    train_size=0.75, test_size=0.25,stratify=Y,random_state=0)
    
    return X_train, X_test, y_train, y_test

def CalcMostFreq(vocabList,fullText):
    '''
    返回前30个高频词
    vocabList：词汇表
    fullText：文本数据样本
    '''
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True) 
    
    return sortedFreq[:30]

def Data_process(dataset,options='minmaxscaler'):
    '''
    对数据进行z-score标准化或者标准化到0到1
    '''
    if options == 'z-score':
        x_ori = np.array(dataset)
        scaler = StandardScaler()
        scaler.fit(x_ori)
        X = scaler.transform(x_ori)
    if options == 'minmaxscaler':
        x_ori = np.array(dataset)
        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit(x_ori)
        X = scaler.transform(x_ori)
    
    return X.tolist(),scaler

def Data_inverse_transform(dataset,scaler):
    x_ori = np.array(dataset)
    X = scaler.inverse_transform(x_ori)
    
    return X.tolist()
    
def TrainNB(trainMatrix,trainCategory):
    '''
    朴素贝叶斯分类器训练函数(此处仅处理两类分类问题)
    trainMatrix:训练集文档矩阵
    trainCategory:训练集每篇文档类别标签
    '''
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pPositive = sum(trainCategory)/float(numTrainDocs)
    #初始化所有词出现数为1，并将分母初始化为2，避免某一个概率值为0
    p0Num = ones(numWords); p1Num = ones(numWords)
    p0Denom = 2.0; p1Denom = 2.0 
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i]) #这里sum(trainMatrix[i])对伯努利贝叶斯方法可能有问题，对词袋模型没问题
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #将结果取自然对数，避免下溢出，即太多很小的数相乘造成的影响
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    
    return p0Vect,p1Vect,pPositive

def TestNB(testMatrix,testCategory,p0Vec,p1Vec,pClass1):
    '''
    朴素贝叶斯分类器测试函数(此处仅处理两类分类问题)
    testMatrix:测试集文档矩阵
    testCategory:测试集每篇文档类别标签
    p0Vec, p1Vec, pClass1:分别对应TrainNB计算得到的3个概率
    '''
    errorCount = 0
    numOfTestSet = len(testMatrix)
    for index in range(numOfTestSet):
        if ClassifyNB(testMatrix[index],p0Vec,p1Vec,pClass1) != testCategory[index]:
            errorCount += 1
    errorRate = float(errorCount)/numOfTestSet
    
    return errorRate
    
def ClassifyNB(testVec,p0Vec,p1Vec,pClass1):
    '''
    分类函数
    testVec:要分类的向量
    p0Vec, p1Vec, pClass1:分别对应TrainNB计算得到的3个概率
    '''
    p1 = sum(testVec * p1Vec) + log(pClass1)
    p0 = sum(testVec * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0

def Gs_PCA(dataset):
    '''
    搜索最优PCA降维参数
    dataset：数据样本
    '''
    X = dataset
    num1 = 0.99
    num2 = 0.98
    num3 = 0.97
    num4 = 0.95
    sum_t = 0
    count = 0
    ret = {}
    pca = PCA(n_components=None)
    pca.fit(X)
    ratios = pca.explained_variance_ratio_
    for ratio in ratios:
        sum_t = sum_t + ratio
        count = count + 1
        if sum_t <= num4:
            ret['95%'] = count
        if sum_t <= num3:
            ret['97%'] = count
        if sum_t <= num2:
            ret['98%'] = count
        if sum_t <= num1:
            ret['99%'] = count
    return pca.n_components_,ret

def Model_PCA(dataset,nums_component):
    '''
    将冗余自由度的数据样本进行降维
    dataset：数据样本
    nums_component：PCA的降维参数
    '''
    X = dataset
    pca = PCA(n_components=nums_component)
    pca.fit(X)
    X_r = pca.transform(X)
    
    return X_r
    
def Gs_DBSCAN_parameter(dataset):
    '''
    利用贪心算法（坐标下降算法），寻找最优的epsilon和min_samples参数
    dataset：数据样本
    '''
    X = dataset
    epsilons = [0.001,0.05,0.06,0.07,0.08,0.1,0.2,0.3,0.5,1,2,3,5]
    min_samples = [2,3,4,5,10,15,20,30,50,70,80,100]
    evalue = []
    mvalue = []
    for epsilon in epsilons:
        clst = DBSCAN(eps = epsilon)
        clst.fit(X)
        if len(set(clst.labels_))>1:
            evalue.append(metrics.silhouette_score(X,clst.labels_,metric='euclidean'))
        else :
            evalue.append(-1)#为了后面的evalue.index(max(evalue))可以找到正确的eindex而补了一个-1的位置
    if len(evalue) == evalue.count(-1):
        raise NameError('empty sequence')
    eindex = evalue.index(max(evalue))
    best_epsilon = epsilons[eindex]
    print "Evaluate Ratio: %s" % evalue
    print "Epsilon Value: %s" % epsilons
    print "============================================="
    for num in min_samples:
        clst = DBSCAN(eps = best_epsilon,min_samples = num)
        clst.fit(X)
        if len(set(clst.labels_))>1:
            mvalue.append(metrics.silhouette_score(X,clst.labels_,metric='euclidean'))
        else :
            mvalue.append(-1)#为了后面的mvalue.index(max(mvalue))可以找到正确的mindex而补了一个-1的位置
    if len(mvalue) == mvalue.count(-1):
        raise NameError('empty sequence')
    mindex = mvalue.index(max(mvalue))
    best_num = min_samples[mindex]
    print "Evaluate Ratio: %s" % mvalue
    print "Min Samples Value: %s" % min_samples
    print "============================================="
    print "Best Epsilon: %s" % best_epsilon
    print "Best Min Samples: %s" % best_num
    
    return best_epsilon,best_num
    
def Model_DBSCAN(dataset,best_epsilon=0.1,best_num=5):
    '''
    使用DBSCAN聚类结果为数据贴标签
    '''
    X = dataset
    
    clst = DBSCAN(eps = best_epsilon, min_samples = best_num)
    clst.fit(X)
    clst_labels = clst.labels_
    if len(set(clst_labels))>1:
        evalue=metrics.silhouette_score(X,clst.labels_,metric='euclidean')
    else:
        evalue="no exception people"
    #输出评价系数
    print "Evaluate Ratio: %s" % evalue
    print "============================================="
    for clst_lab in set(clst_labels):
        print "Number of the %s class: %s" % (clst_lab,list(clst_labels).count(clst_lab))
    print "============================================="
    print "Number of the labels: %s" % len(clst_labels)
    
    return clst_labels

def Show_data(vocabset,dataset,labels):
    '''
    在已经标签的数据中，将每个大于0的词输出来
    vocabset：词汇表
    dataset：词频向量数据集
    labels：词频向量的标签
    '''
    show_data = []
    num_vocabset = len(vocabset)
    num_dataset = len(dataset)
    for data_index in range(num_dataset):
        tem = []
        tem_dict = {}
        for word_index in range(num_vocabset):
            if dataset[data_index][word_index] > 0.1:
                tem.extend([vocabset[word_index]])
        tem_dict['class %s:'%labels[data_index]] = tem
        show_data.append(tem_dict)
        
    return show_data

def Gs_auto_encoder_parameter(dataset,pca_dim=30):
    '''
    利用贪心算法（坐标下降算法），寻找单隐层自编码器最优的encoding_dim参数
    PS: loss='mean_squared_error'是因为dataset中数据中的值不是0或1
    '''
    X = np.array(dataset)
    dim_data = X.shape[1]
    X_train, X_test = train_test_split(X,train_size=0.75, test_size=0.25,random_state=0)
    
    if pca_dim > 3:
        encoding_dim = [pca_dim-3,pca_dim-1,pca_dim,pca_dim+1,pca_dim+3]
    else:
        encoding_dim = [pca_dim,pca_dim+1,pca_dim+3]
    input_img = Input(shape=(dim_data,))
    score =[]
    
    for en_dim in encoding_dim:
        encoded = Dense(en_dim, activation='relu',activity_regularizer=regularizers.l1(10e-5))(input_img)
        #因为输入的数据取值范围是0到1，所以用sigmoid可以很好的对应
        decoded = Dense(dim_data, activation='sigmoid')(encoded)
        
        autoencoder = Model(input=input_img, output=decoded)
        encoder = Model(input=input_img, output=encoded)
        
        encoded_input = Input(shape=(en_dim,))
        decoder_layer = autoencoder.layers[-1]
        decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
        
        autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
        autoencoder.fit(X_train, X_train,
                epochs=80,
                batch_size=64)
        score.append(autoencoder.evaluate(X_test, X_test, batch_size=64))
        
    sindex = score.index(min(score))
    best_encoding_dim = encoding_dim[sindex]
    print "Evaluate Ratio: %s" % score
    print "Encoding_dim Value: %s" % encoding_dim
    print "============================================="
    print "Best encoding_dim: %s" % best_encoding_dim
    
    return best_encoding_dim
    
def Model_auto_encoder(dataset,best_encoding_dim):
    '''
    建立单隐层的自编码器
    使用详情：https://github.com/MoyanZitto/keras-cn/blob/master/docs/legacy/blog/autoencoder.md 
    '''
    X = np.array(dataset)
    dim_data = X.shape[1]
    en_dim = best_encoding_dim
    input_img = Input(shape=(dim_data,))
    #activity_regularizer=regularizers.l1(10e-5),对隐层单元施加稀疏性约束的话，会得到更为紧凑的表达
    encoded = Dense(en_dim, activation='relu',activity_regularizer=regularizers.l1(10e-5))(input_img)
    #因为输入的数据取值范围是0到1，所以用sigmoid可以很好的对应
    decoded = Dense(dim_data, activation='sigmoid')(encoded)
        
    autoencoder = Model(input=input_img, output=decoded)
    encoder = Model(input=input_img, output=encoded)
        
    encoded_input = Input(shape=(en_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
    #如何选择loss函数 binary_crossentropy mean_squared_error mean_absolute_error
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
    autoencoder.fit(X, X,
                epochs=80,
                batch_size=64)
    encoded_imgs = encoder.predict(X)
    X_decoded = decoder.predict(encoded_imgs)
    
    return X_decoded
    
def Model_deep_auto_encoder(dataset):
    '''
    建立多隐层的自编码器
    使用详情：https://github.com/MoyanZitto/keras-cn/blob/master/docs/legacy/blog/autoencoder.md 
    '''
    X = np.array(dataset)
    dim_data = X.shape[1]
    X_train, X_test = train_test_split(X,train_size=0.75, test_size=0.25,random_state=0)
    
    input_img = Input(shape=(dim_data,))
    batch_size = 64
    epochs = 80
    #activity_regularizer=regularizers.l1(10e-5),对隐层单元施加稀疏性约束的话，会得到更为紧凑的表达
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    #因为输入的数据取值范围是0到1，所以用sigmoid可以很好的对应
    decoded = Dense(dim_data, activation='sigmoid')(decoded)
        
    autoencoder = Model(input=input_img, output=decoded)
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')#mean_absolute_error
    autoencoder.fit(X_train, X_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(X_test, X_test))
    X_decoded = autoencoder.predict(X)
    
    return X_decoded

def Model_deep_auto_encoder_noisy(dataset):
    '''
    建立去噪多隐层的自编码机
    使用详情：https://github.com/MoyanZitto/keras-cn/blob/master/docs/legacy/blog/autoencoder.md 
    '''
    X = np.array(dataset)
    dim_data = X.shape[1]
    X_train, X_test = train_test_split(X,train_size=0.75, test_size=0.25,random_state=0)
    
    noise_factor = 0.01
    X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape) 
    X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape) 
    #因为dataset中的数据在预处理中已经被归一化到0到1之间，所以为了之后的sigmoid层(输出为0到1之间)，即使加了噪声，值也要在0到1之间
    X_train_noisy = np.clip(X_train_noisy, 0., 1.)
    X_test_noisy = np.clip(X_test_noisy, 0., 1.)

    input_img = Input(shape=(dim_data,))
    batch_size = 64
    epochs = 80
    #activity_regularizer=regularizers.l1(10e-5),对隐层单元施加稀疏性约束的话，会得到更为紧凑的表达
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    #因为输入的数据取值范围是0到1，所以用sigmoid可以很好的对应
    decoded = Dense(dim_data, activation='sigmoid')(decoded)
        
    autoencoder = Model(input=input_img, output=decoded)
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
    autoencoder.fit(X_train_noisy, X_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(X_test_noisy, X_test))
    X_decoded = autoencoder.predict(X)
    
    return X_decoded
    
def Model_variational_autoencoder(dataset):
    '''
    变分自编码器
    使用详情：https://github.com/MoyanZitto/keras-cn/blob/master/docs/legacy/blog/autoencoder.md 
    '''
    X = np.array(dataset)
    original_dim = X.shape[1]
    X_train, X_test = train_test_split(X,train_size=0.75, test_size=0.25,random_state=0)
    
    batch_size = 64
    latent_dim = 30
    intermediate_dim = 128
    epochs = 80
    epsilon_std = 1.0
    # 建立编码网络，将输入影射为隐分布的参数
    x = Input(shape=(original_dim,))
    h = Dense(intermediate_dim, activation='relu')(x)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    # 从这些参数确定的分布中采样，这个样本相当于之前的隐层值
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon
    
    z = keras_Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    # 采样得到的点映射回去重构原输入
    decoder_h = Dense(intermediate_dim, activation='relu')
    #因为输入的数据取值范围是0到1，所以用sigmoid可以很好的对应
    decoder_mean = Dense(original_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)
    # 构建VAE模型
    vae = Model(x, x_decoded_mean)
    # 使用端到端的模型训练，损失函数是一项重构误差，和一项KL距离
    #xent_loss = original_dim * keras_metrics.binary_crossentropy(x, x_decoded_mean)
    xent_loss = original_dim * keras_metrics.mean_squared_error(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)

    vae.add_loss(vae_loss)
    vae.compile(optimizer='rmsprop')
    vae.summary()
    
    vae.fit(X_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, None))
    
    X_encoded = vae.predict(X)
    
    return X_encoded
    
    