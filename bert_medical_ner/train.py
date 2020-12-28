# -*- coding: utf-8 -*-

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from utils import load_vocab, load_data, recover_label, get_ner_fmeasure, save_model, load_model
from constants import *
from model import BERT_LSTM_CRF
import os


print('device',device)
# if torch.cuda.is_available():
#     device = torch.device("cuda", 2)
#     print('device',device)
#     use_cuda = True
# else:
#     device = torch.device("cpu")
#     use_cuda = False

vocab = load_vocab(vocab_file)
vocab_reverse = {v:k for k, v in vocab.items()}

print('max_length',max_length)


train_data = load_data(train_file, max_length=max_length, label_dic=l2i_dic, vocab=vocab)
train_ids = torch.LongTensor([temp.input_id for temp in train_data[1500:]])
train_masks = torch.LongTensor([temp.input_mask for temp in train_data[1500:]])
train_tags = torch.LongTensor([temp.label_id for temp in train_data[1500:]])
train_lenghts = torch.LongTensor([temp.lenght for temp in train_data[1500:]])
train_dataset = TensorDataset(train_ids, train_masks, train_tags,train_lenghts)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

dev_data = load_data(dev_file, max_length=max_length, label_dic=l2i_dic, vocab=vocab)
dev_ids = torch.LongTensor([temp.input_id for temp in dev_data[:1500]])
dev_masks = torch.LongTensor([temp.input_mask for temp in dev_data[:1500]])
dev_tags = torch.LongTensor([temp.label_id for temp in dev_data[:1500]])
dev_lenghts = torch.LongTensor([temp.lenght for temp in dev_data[:1500]])
dev_dataset = TensorDataset(dev_ids, dev_masks, dev_tags,dev_lenghts)
dev_loader = DataLoader(dev_dataset, shuffle=True, batch_size=batch_size)


test_data = load_data(test_file, max_length=max_length, label_dic=l2i_dic, vocab=vocab)
test_ids = torch.LongTensor([temp.input_id for temp in test_data])
test_masks = torch.LongTensor([temp.input_mask for temp in test_data])
test_tags = torch.LongTensor([temp.label_id for temp in test_data])
test_lenghts = torch.LongTensor([temp.lenght for temp in test_data])


test_dataset = TensorDataset(test_ids, test_masks, test_tags,test_lenghts)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

def every(real_entity_mark, predict_entity_mark):
    true_entity_mark = dict()
    key_set = real_entity_mark.keys() & predict_entity_mark.keys()

    for key in key_set:
        real_entity = real_entity_mark.get(key)
        predict_entity = predict_entity_mark.get(key)
        if tuple(real_entity) == tuple(predict_entity):
            true_entity_mark.setdefault(key, real_entity)

    real_entity_num = len(real_entity_mark)
    predict_entity_num = len(predict_entity_mark)
    true_entity_num = len(true_entity_mark)
    if predict_entity_num==0 or real_entity_num==0 or true_entity_num==0:
        return 0,0,0

    precision = true_entity_num / predict_entity_num
    recall = true_entity_num / real_entity_num
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def split_entity(label_sequence):
    entity_mark = dict()
    entity_pointer = None
    #print(label_sequence)
    for index, label in enumerate(label_sequence):
        if label.startswith('B'):
            category = label.split('_')[1]
            entity_pointer = (index, category)
            entity_mark.setdefault(entity_pointer, [label])
        elif label.startswith('M'):
            if entity_pointer is None: continue
            if entity_pointer[1] != label.split('_')[1]: continue
            entity_mark[entity_pointer].append(label)
        elif label.startswith('E'):
            if entity_pointer is None: continue
            if entity_pointer[1] != label.split('_')[1]: continue
            entity_mark[entity_pointer].append(label)
        else:
            entity_pointer = None
    return entity_mark
def evaluate_tag( real_, predict_):


    real_entity_mark = split_entity(real_)
    predict_entity_mark = split_entity(predict_)
    real_entity_mark_dis = {}
    predict_entity_mark_dis = {}
    real_entity_mark_bod = {}
    predict_entity_mark_bod = {}
    #print("real",real_entity_mark)

    for key, value in real_entity_mark.items():
        # print(key)
        if key[1] == "dis":
            real_entity_mark_dis.update({key: value})

        elif key[1] == "bod":
            real_entity_mark_bod.update({key: value})
    #print(real_entity_mark_dis)
    for key, value in predict_entity_mark.items():
        if key[1] == "dis":
            predict_entity_mark_dis.update({key: value})

        elif key[1] == "bod":
            predict_entity_mark_bod.update({key: value})

    #print("pre",predict_entity_mark_pro)
    #print("real",real_entity_mark_pro)
    pre_dis, recall_dis, f_dis = every(real_entity_mark_dis, predict_entity_mark_dis)
    pre_bod, recall_bod, f_bod = every(real_entity_mark_bod, predict_entity_mark_bod)

    print("疾病：")
    print("准确率：", pre_dis)
    print("召回率：", recall_dis)
    print("f1:", f_dis)

    print("身体：")
    print("准确率：", pre_bod)
    print("召回率：", recall_bod)
    print("f1:", f_bod)

# test 函数
def evaluate_all(real_, predict_):
    #print(real_)


    real_entity_mark = split_entity(real_)
    predict_entity_mark = split_entity(predict_)
    #print(predict_entity_mark)

    true_entity_mark = dict()
    key_set = real_entity_mark.keys() & predict_entity_mark.keys()
    for key in key_set:
        real_entity = real_entity_mark.get(key)
        predict_entity = predict_entity_mark.get(key)
        if tuple(real_entity) == tuple(predict_entity):
            true_entity_mark.setdefault(key, real_entity)

    real_entity_num = len(real_entity_mark)
    predict_entity_num = len(predict_entity_mark)
    true_entity_num = len(true_entity_mark)
    if predict_entity_num==0 or real_entity_num==0 or true_entity_num==0:
        print("准确率为0")
        f1=0

    else:
        precision = true_entity_num / predict_entity_num
        recall = true_entity_num / real_entity_num
        f1 = 2 * precision * recall / (precision + recall)
        print("总体的结果")
        print("准确率：", precision)
        print("召回率：", recall)
        print("f1:", f1)
    return f1


######测试函数
def evaluate(medel, dev_loader):
    medel.eval()
    pred = []
    gold = []
    pred_test = []
    pred_list=[]
    gold_list=[]

    print('evaluate')
    with torch.no_grad():
        for i, dev_batch in enumerate(dev_loader):
            sentence, masks, tags , lengths = dev_batch
            sentence, masks, tags, lengths = Variable(sentence), Variable(masks), Variable(tags), Variable(lengths)
            if use_cuda:
                sentence = sentence.to(device)
                masks = masks.to(device)
                tags = tags.to(device)

            predict_tags = medel(sentence, masks)
            loss = model.neg_log_likelihood_loss(sentence, masks, tags)

            pred.extend([t for t in predict_tags.tolist()])
            gold.extend([t for t in tags.tolist()])

        pred_label,gold_label = recover_label(pred, gold, l2i_dic,i2l_dic)
        print('dev loss {}'.format(loss.item()))
        pred_label_1 = [t[1:] for t in pred_label]
        gold_label_1 = [t[1:] for t in gold_label]
        for i in pred_label_1:
            for j in range(len(i)-1):
                pred_list.append(i[j])

            pred_list.append(i[len(i)-1])
        for i in gold_label_1:
            for j in range(len(i) - 1):
                gold_list.append(i[j])

            gold_list.append(i[len(i) - 1])



       # fw = open('data/predict_result'+str(float('%.3f'%dev_f))+'bert.txt','w')

      
      #  print(len(pred_list), pred_list)
      #  print(len(gold_list),gold_list)
        print("验证集的结果")
        f=evaluate_all(gold_list, pred_list)
        
        evaluate_tag(gold_list, pred_list)
    return f

# test 函数
def evaluate_test(medel,test_loader,dev_f):
    medel.eval()
    pred = []
    gold = []
    pred_list=[]
    gold_list=[]
    pred_final=[]
    gold_final=[]
  
    print('test')
    with torch.no_grad():
        for i, dev_batch in enumerate(test_loader):
            sentence, masks, tags, lengths = dev_batch
            sentence, masks, tags , lengths = Variable(sentence), Variable(masks), Variable(tags),Variable(lengths)
            if use_cuda:
                sentence = sentence.to(device)
                masks = masks.to(device)
                tags = tags.to(device)
            predict_tags = medel(sentence, masks)

            pred.extend([t for t in predict_tags.tolist()])
            gold.extend([t for t in tags.tolist()])

        pred_label, gold_label = recover_label(pred, gold, l2i_dic,i2l_dic)
        pred_label_2 = [t[1:] for t in pred_label]
        gold_label_2 = [t[1:] for t in gold_label]
        fw = open('data/predict_result'+str(float('%.3f'%dev_f))+'bert.txt','w')
        for i in pred_label_2:
            for j in range(len(i)-1):
                fw.write(i[j])
                fw.write(' ')
            fw.write(i[len(i)-1])
            fw.write('\n')
        for i in pred_label_2:
            for j in range(len(i)-1):
                pred_list.append(i[j])

            pred_list.append(i[len(i)-1])
        for i in gold_label_2:
            for j in range(len(i) - 1):
                gold_list.append(i[j])

            gold_list.append(i[len(i) - 1])



       # fw = open('data/predict_result'+str(float('%.3f'%dev_f))+'bert.txt','w')

      
 #       print(len(pred_list), pred_list)
#        print(len(gold_list),gold_list)
        print("测试集的结果")
        _=evaluate_all(gold_list, pred_list)
        evaluate_tag(gold_list, pred_list)
    



model = BERT_LSTM_CRF('data/chinese_wwm_pytorch', tagset_size, 768, 200, 2,
                      dropout_ratio=0.5, dropout1=0.5, use_cuda = use_cuda)

if use_cuda:
    model.to(device)

optimizer = getattr(optim, 'Adam')
optimizer = optimizer(model.parameters(), lr=0.000005, weight_decay=0.00005)
#model.load_state_dict(torch.load('data/model/711.pkl'))
best_f = -100
model_name = save_model_dir + '0715' + str(float('%.3f' % best_f)) + ".pkl"
print(model_name)

for epoch in range(epochs):
    print('epoch: {}，train'.format(epoch))
    for i, train_batch in enumerate(tqdm(train_loader)):
        sentence, masks, tags , lengths= train_batch

        sentence, masks, tags , lengths = Variable(sentence), Variable(masks), Variable(tags), Variable(lengths)

        if use_cuda:
            sentence = sentence.to(device)
            masks = masks.to(device)
            tags = tags.to(device)
        model.train()
        optimizer.zero_grad()
        loss = model.neg_log_likelihood_loss(sentence, masks, tags)
        loss.backward()
        optimizer.step()

    print('epoch: {}，train loss: {}'.format(epoch, loss.item()))
    f=evaluate(model,dev_loader)


    if f > best_f:
        best_f = f
        evaluate_test(model, test_loader, loss.item())
        model_name = save_model_dir + 'new0710' + str(float('%.3f' % best_f)) + ".pkl"
        torch.save(model.state_dict(), model_name)











