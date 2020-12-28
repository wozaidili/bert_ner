from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from prettytable import PrettyTable
from utils import load_vocab, load_data, recover_label, get_ner_fmeasure, save_model, load_model
from constants import *
from model import BERT_LSTM_CRF
vocab = load_vocab(vocab_file)
vocab_reverse = {v:k for k, v in vocab.items()}

model = BERT_LSTM_CRF('data/chinese_wwm_pytorch', tagset_size, 768, 200, 2,
                      dropout_ratio=0.5, dropout1=0.5, use_cuda = use_cuda)
if use_cuda:
    model.to(device)
#file_test_new='./data/test_data_0712.txt'
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

    precision = round(true_entity_num / predict_entity_num,3)
    recall = round(true_entity_num / real_entity_num,3)
    f1 = round(2 * precision * recall / (precision + recall),3)

    return precision, recall, f1


def split_entity(label_sequence):
    entity_mark = dict()
    entity_pointer = None
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


    for key, value in real_entity_mark.items():
        if key[1] == "dis":
            real_entity_mark_dis.update({key: value})
        elif key[1] == "bod":
            real_entity_mark_bod.update({key: value})

    for key, value in predict_entity_mark.items():
        if key[1] == "dis":
            predict_entity_mark_dis.update({key: value})

        elif key[1] == "bod":
            predict_entity_mark_bod.update({key: value})

    pre_dis, recall_dis, f_dis = every(real_entity_mark_dis, predict_entity_mark_dis)

    pre_bod, recall_bod, f_bod = every(real_entity_mark_bod, predict_entity_mark_bod)

    result=dict()
    result["疾病"]=[pre_dis,recall_dis,f_dis]
    result["身体"]=[pre_bod,recall_bod,f_bod]

    return result

    # print("疾病：")
    # print("准确率：", pre_dis)
    # print("召回率：", recall_dis)
    # print("f1:", f_dis)
    # print("临床表现：")
    # print("准确率：", pre_sym)
    # print("召回率：", recall_sym)
    # print("f1:", f_sym)
    # print("医疗程序：")
    # print("准确率：", pre_pro)
    # print("召回率：", recall_pro)
    # print("f1:", f_pro)
    # print("医疗设备：")
    # print("准确率：", pre_equ)
    # print("召回率：", recall_equ)
    # print("f1:", f_equ)
    # print("身体：")
    # print("准确率：", pre_bod)
    # print("召回率：", recall_bod)
    # print("f1:", f_bod)
    # print("药物：")
    # print("准确率：", pre_dru)
    # print("召回率：", recall_dru)
    # print("f1:", f_dru)
    # print("医学检验项目：")
    # print("准确率：", pre_ite)
    # print("召回率：", recall_ite)
    # print("f1:", f_ite)
    # print("微生物类：")
    # print("准确率：", pre_mic)
    # print("召回率：", recall_mic)
    # print("f1:", f_mic)
    # print("科室：")
    # print("准确率：", pre_dep)
    # print("召回率：", recall_dep)
    # print("f1:", f_dep)
# test 函数
def evaluate_all(real_, predict_):
    real_entity_mark = split_entity(real_)
    predict_entity_mark = split_entity(predict_)

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

    precision = round(true_entity_num / predict_entity_num,3)
    recall = round(true_entity_num / real_entity_num,3)
    f1 = round(2 * precision * recall / (precision + recall),3)
    result_all=dict()
    result_all["总体"]=[precision,recall,f1]
    # print("总体的结果")
    # print("准确率：", precision)
    # print("召回率：", recall)
    # print("f1:", f1)
    return result_all
def evaluate_test(medel,test_loader):
    model.load_state_dict(torch.load(medical_tool_model, map_location=device))
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
        for i in pred_label_2:
            for j in range(len(i)-1):
                pred_list.append(i[j])

            pred_list.append(i[len(i)-1])
        for i in gold_label_2:
            for j in range(len(i) - 1):
                gold_list.append(i[j])

            gold_list.append(i[len(i) - 1])

        for i in pred_list:
            
            pred_final.append(i)
        for i in gold_list:
           
            gold_final.append(i)
        result_all=evaluate_all(gold_final, pred_final)
        result_tag=evaluate_tag(gold_final, pred_final)
        table = PrettyTable(['实体类别','准确率','召回率','F1值'])
        table.add_row(['疾病',result_tag["疾病"][0],result_tag["疾病"][1],result_tag["疾病"][2]])
        table.add_row(['身体',result_tag["身体"][0],result_tag["身体"][1],result_tag["身体"][2]])

        print(table)

evaluate_test(model,test_loader)
