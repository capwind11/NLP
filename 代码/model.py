# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import numpy as np
import os
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
import nltk
# 有gpu的情况下使用gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
local = torch.device('cpu')
# 为了防止GPU内存溢出，将训练后的数据转到CPU存储
# device = torch.device('cpu')

# beam_search搜索
def beam_search(beam_width,candidate,score,prob,index,h):
    prob = prob+score
    d = torch.sort(prob.reshape(-1),0,True)
    new_candidate = []
    for i in range(beam_width):
        score[i]=d[0][i]
        r = (d[1][i])//beam_width
        l = (d[1][i])%beam_width
        h[i] = h[r]
        new_candidate.append(candidate[i]+[index[r][l]])
    return (new_candidate,score,h)

# 建立词典，包括从单词到序号以及序号到单词的映射
class voc_dict():
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_words(self,word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

# 建立数据集，生成模型的输入数据
class build_dataset():
    # 初始化词典，同时添加起始和结束符
    def __init__(self):
        self.dictionary = voc_dict()
        self.dictionary.add_words('bos')
        self.dictionary.add_words('eos')
    
    def size(self):
        return len(self.dictionary.word2idx)

    # 对指定文件的数据进行预处理，并输出步长为time_steps的训练和测试数据
    def get_data(self,file,time_steps):
        dataset = []
        self.dictionary.add_words('padding')
        with open(file,'r',encoding='utf-8') as f:
            content = f.readlines()
            for line in content[:100]:
                words = line.strip().split()
                if len(words)>time_steps-2:
                    words = words[0:time_steps-2]
                words = ['bos']+words+['eos']
                idx = []
                for word in words:
                    self.dictionary.add_words(word)
                while (len(words)<time_steps):
                    words= words + ['padding']
                for word in words[0:time_steps]:
                    idx.append(self.dictionary.word2idx[word])
                dataset.append(idx)
        dataset = np.array(dataset)
        return dataset

# 为方便观察结果，可以将结果序列转换成中英文的句子
def seq2words(dict,text):
    result = ''
    for i in text:
        try:
            i = i.numpy() # 如果是torch类型，则先转换成numpy
        except:
            pass
        for j in i:
            if j!=2:
                result+=dict.dictionary.idx2word[j]+' '
            if j==1:
                break
        result+='\n'
    return result

# attention层的定义，可以在decoder模型中直接引入
class attention(nn.Module):
    def __init__(self,hidden_size):
        super(attention, self).__init__()
        self.hidden_size = hidden_size
        # 要训练的网络总共有两个全连接层
        # 获取score(ht,hs) 的全连接层，训练参数为W
        self.score = nn.Linear(hidden_size,hidden_size)
        # 获取attention vector tanh(Wc[ct;ht])的训练参数Wc
        self.attention_vetcor = nn.Linear(2*hidden_size,hidden_size)
        
    def forward(self,hs,ht):
        # hs为encoder隐藏层所有time_step输出的结果，ht为decoder中一个time_step隐藏层输出结果
        # hs的维度大小:(bs,seq_len,hidden_size) ht的维度大小(batch_size,1,hidden_size)
        score = self.score(hs)
        # (batch_size,seq_len,hidden_size)->(batch_size,seq_len,hidden_size)
        score = score.transpose(1,2)
        # 进行转置得到(batch_size,hidden_size,seq_len)
        score = torch.bmm(ht,score)
        # (batch_size,1,hidden_size)*(batch_size,hidden_size,seq_len) = (batch_size,1,seq_len)
        self.weight = F.softmax(score,dim=2)
        # 对第二个维度进行softmax，使每个batch的每个time_step权重之和为1
        ct = torch.bmm(self.weight,hs)
        # (batch_size,1,seq_len)*(bs,seq_len,hidden_size) = (batch_size,1,hidden_size)
        h = torch.cat((ct,ht),2)
        # 拼接操作后得到 (batch_size,1,2*hidden_size)
        output = self.attention_vetcor(h)
        # (batch_size,1,2*hidden_size)->(batch_size,1,hidden_size)
        output = F.torch.tanh(output)
        return output
        
# 编码模型
class encoder(nn.Module):
    def __init__(self,input_size,embed_size, hidden_size):
        # input_size为中文单词总数，embed_size为embedding压缩后向量空间维度
        super(encoder,self).__init__()
        self.hidden_size = hidden_size
        # 将input_size的one-hot编码压缩到embed_size的向量空间
        self.embed = nn.Embedding(input_size, embed_size)
        # 双向lstm将embedding层输出卷积为 hidden_size的输出
        self.lstm = nn.LSTM(embed_size, hidden_size,batch_first=True, bidirectional=True)

    def forward(self, x, h=None):
        batch_size = x.size()[0]
        # (batch_size,seq_len,voc_size)->(batch_size,seq_len,embed_size)
        x = self.embed(x)
        # 初始化输入隐藏层的状态为全0
        if h is None:
            h_0 = x.data.new(2, batch_size, self.hidden_size).fill_(0).float()
            c_0 = x.data.new(2, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0, c_0 = h
        # (batch_size,seq_len, embed_size) -> (batch_size,seq_len, hidden_size*2)
        hidden_out,h = self.lstm(x,(h_0, c_0) )
        output = hidden_out[:, :, :self.hidden_size] + hidden_out[:, :, self.hidden_size:]
        # (batch_size,seq_len, embed_size*2) -> (batch_size,seq_len, hidden_size)
        return output,h
     
class decoder(nn.Module):
    def __init__(self,input_size,embed_size, hidden_size):
        super(decoder,self).__init__()
        self.embed = nn.Embedding(input_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size,batch_first=True)
        # 引入attention层
        self.attention_layer = attention(hidden_size)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x, hs, hm):
        # hs为encoder隐藏层输出，hm为前一单元的隐藏层状态传递
        x = self.embed(x)
        # (batch_size,1,voc_size)->(batch_size,1,embed_size)
        hidden_out,h = self.lstm(x,hm) 
        # hidden_out:(batch_size,1,hidden_size)
        # h为一个包含两个(1,batch_size,hidden_size)的元组
        hidden_out = self.attention_layer(hs,hidden_out)
        # hidden_out: (batch_size,1,hidden_size)
        out = self.linear(hidden_out)
        # (batch_size,1,hidden_size)->(batch_size,1,voc_size)
        return out,h

# 建立中文字典和英文字典
zh_dict = build_dataset()
data_source = torch.LongTensor(zh_dict.get_data('source.txt',20))   
# print(zh_dict.size())

en_dict = build_dataset()
data_target = torch.LongTensor(en_dict.get_data('target.txt',20))
# print(en_dict.size())

# 超参数设置 #
input_size = zh_dict.size()
output_size = en_dict.size()
embed_size = 1024
hidden_size = 512
num_epochs = 100 #训练轮数
batch_size = 50  #一个batch的大小
time_steps = 20  #序列长度
lr = 0.002  #学习率
teacher_force_rate = 0.5 #强制学习率
beam_width = 3 #beam_search分支大小

#定义网络
encoding = encoder(input_size, embed_size, hidden_size).to(device)
print(encoding)
decoding = decoder(output_size, embed_size, hidden_size).to(device)
print(decoding)

# 交叉熵忽略index为2，即padding的损失
criterion = nn.CrossEntropyLoss(ignore_index=2)
encoder_optimizer = torch.optim.Adam(encoding.parameters(),lr=0.003)
decoder_optimizer = torch.optim.Adam(decoding.parameters(),lr=0.003)

# 划分训练集和数据集
train_source = data_source[:8000,:] 
train_target = data_target[:8000,:]
test_source = data_source[8000:10000,:] 
test_target = data_target[8000:10000,:]

loss_list = [] #保存loss
for epoch in range(num_epochs): 
    for i in range(0,train_source.size(0)//batch_size-1): 
        input = train_source[i*batch_size:(i+1)*batch_size,:].to(device) 
        target = train_target[i*batch_size:(i+1)*batch_size,:] 
        input.to(local)
        # 将训练完的数据放回cpu防止内存耗尽
        # 对双向lstm的输出结果进行处理，会输出两个单元的隐藏层状态
        hs, (h_n, h_c) = encoding(input) 
        h_n = (h_n[0] + h_n[1]).unsqueeze(0)
        h_c = (h_c[0] + h_c[1]).unsqueeze(0)
        h_c = (h_n,h_c)
        # 开始迭代训练decoder
        x = target[:,[0]].to(device) 
        result,h_c= decoding(x,hs,h_c)
        # 得到当前预测结果
        x = torch.max(result, 2)[1]
        for j in range(1,target.size()[1]-1):
            use_teacher_forcing = True if random.random() < teacher_force_rate else False
            # 随机输入正确标签
            if use_teacher_forcing:
                x = target[:,[j]].to(device) 
            output,h_c = decoding(x,hs,h_c)
            x = torch.max(output, 2)[1]
            # 得到当前结果
            result=torch.cat((result,output),1)
            # 将每一步结果拼接
        result = result.reshape(result.size(0)*result.size(1),-1)
        # 将预测结果和目标结果对齐计算交叉熵
        loss = criterion(result,target[:,1:].to(device).reshape(-1)) 
        if i==0:
            total_loss = loss
        else:
            total_loss+=loss
        # 反向传播
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        
    total_loss/=train_source.size(0)//batch_size-1
    loss_list.append(total_loss)
#     torch.cuda.empty_cache()
    if epoch%5 == 0:
        # 每5轮查看一次测试集上的预测结果
        target = test_target
        input = test_source.to(device)
        hs, (h_n, h_c) = encoding(input) 
        h_n = (h_n[0] + h_n[1]).unsqueeze(0)
        h_c = (h_c[0] + h_c[1]).unsqueeze(0)
        h_c = (h_n,h_c)
        input.cpu()
        x = target[:,[0]].to(device) 
        result,h_c= decoding(x,hs,h_c)
        result.cpu()
        x = torch.max(result, 2)[1]
        for i in range(1,target.size()[1]-1):
            output,h_c = decoding(x,hs,h_c)
            x = torch.max(output, 2)[1]
#             print(h_c.size())
            output.cpu()
            result=torch.cat((result,output),1)      
        result = result.reshape(result.size(0)*result.size(1),-1)
        pred_y = torch.max(result, 1)[1].data.cpu().numpy().reshape(-1,19)
        target = target[:,1:].data.cpu().numpy().reshape(-1)
#         print(seq2words(en_dict,target[0:19].reshape(-1,19)))
        pred_y = pred_y.reshape(-1)
        f = open('result/'+str(epoch)+'.txt','w',encoding = 'utf-8')
        pred_y = pred_y.reshape(-1,19)
        target = target.reshape(-1,19)
        BLEUscore = 0
        for i in range(pred_y.shape[0]):
            result = seq2words(en_dict,pred_y[i].reshape(1,-1))
            hypothesis=result.split(' ')
            reference = seq2words(en_dict,target[i].reshape(1,-1)).split(' ')
            BLEUscore += nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights = [1]) 
            f.write(result+'\n')
        f.close()

        BLEUscore = BLEUscore /pred_y.shape[0]
        print('Epoch: ', epoch, '| train loss: %.4f' % total_loss.data.cpu().numpy())#, '| test accuracy: %.2f ' % accuracy,'|BLEUscore:  %.3f'%round(BLEUscore,8))

plt.plot(loss_list, label='training loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()

# encoding.cpu()
# decoding.cpu()
# torch.save(encoding, 'encode.pkl')
# torch.save(decoding, 'decode.pkl')

# 训练完成后利用beam_search策略对整个测试集进行测试
BLEUscore = 0
pred_seq = ''
beam_width = 3
f = open('beam_result.txt','w',encoding = 'utf-8')
for n in range(test_target.size()[0]):
    target = test_target[[n],:] 
    input = test_source[[n],:].to(device)
    hs, (h_n, h_c) = encoding(input) 
    h_n = (h_n[0] + h_n[1]).unsqueeze(0)
    h_c = (h_c[0] + h_c[1]).unsqueeze(0)
    h_c = (h_n,h_c)
    input.cpu()
    x = target[:,[0]].to(device) 
    result,h= decoding(x,hs,h_c)
    h = [h]*beam_width
    tmp =torch.sort(result, 2,True)[1][:,:,:beam_width].reshape(-1)
    beam = [ [i] for i in tmp]
    # 第一个时间步抽取前beam_width大概率的翻译结果
    score = torch.sort(result, 2)[0][:,:,:beam_width].reshape(beam_width,-1)
    for i in range(1,target.size()[1]-1):
        x = [j[-1] for j in beam]
        out = []
        for j in range(beam_width):
            input = x[j].long().reshape(1,1)
            result,h[j]= decoding(input,hs,h[j])
            out.append(result)
        out = torch.cat([i for i in out],dim = 1)
        prob = torch.sort(out,2,True)[0][:,:,:beam_width].reshape(-1,beam_width)
        index = torch.sort(out,2,True)[1][:,:,:beam_width].reshape(-1,beam_width)
        (beam,score,h) = beam_search(beam_width,beam,score,prob,index,h)
    index = torch.max(score, 0)[1].data.cpu().numpy()
    pred_seq=''
    for j in beam[0]:
        #j = j.numpy()
        if j!=2:
            if en_dict.dictionary.idx2word[int(j)] == 'eos':
                break
            pred_seq+=en_dict.dictionary.idx2word[int(j)]+' '
    hypothesis =pred_seq.split(' ')
    reference = seq2words(en_dict,target.reshape(1,-1))
    f.write(str(n)+' 中文'+seq2words(zh_dict,test_source[[n],:].reshape(1,-1))+'预测翻译: '+pred_seq+'正确翻译:'+reference+'\n')
    reference=reference.split(' ')
    BLEUscore += nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights = [1]) 
f.close()
BLEUscore = BLEUscore/train_target.size()[0]
print('beam search: '+str(BLEUscore))

