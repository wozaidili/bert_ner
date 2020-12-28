# coding:utf-8
import codecs
import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import json
from utils import load_vocab
from constants import *
from model import BERT_LSTM_CRF
import os


class medical_ner(object):
    def __init__(self):
        self.NEWPATH = medical_tool_model
        self.vocab = load_vocab(vocab_file)
        self.vocab_reverse = {v: k for k, v in self.vocab.items()}

        self.model = BERT_LSTM_CRF('data/chinese_wwm_pytorch', tagset_size, 768, 200, 2,
                              dropout_ratio=0.5, dropout1=0.5, use_cuda=use_cuda)

        if use_cuda:
            self.model.to(device)

    def from_input(self, input_str):
        raw_text = []
        textid = []
        textmask = []
        textlength = []
        text = ['[CLS]'] + [x for x in input_str] + ['[SEP]']
        raw_text.append(text)
        cur_len = len(text)
        raw_textid = [self.vocab[x] for x in text] + [0] * (max_length - cur_len)
        textid.append(raw_textid)
        raw_textmask = [1] * cur_len + [0] * (max_length - cur_len)
        textmask.append(raw_textmask)
        textlength.append([cur_len])
        textid = torch.LongTensor(textid)
        textmask = torch.LongTensor(textmask)
        textlength = torch.LongTensor(textlength)
        return raw_text, textid, textmask, textlength

    def from_txt(self, input_path):
        raw_text = []
        textid = []
        textmask = []
        textlength = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if len(line.strip())==0:
                    continue
                if len(line) > 198:
                    line = line[:198]
                temptext = ['[CLS]'] + [x for x in line[:-1]] + ['[SEP]']
                cur_len = len(temptext)
                raw_text.append(temptext)

                tempid = [self.vocab[x] for x in temptext[:cur_len]] + [0] * (max_length - cur_len)
                textid.append(tempid)
                textmask.append([1] * cur_len + [0] * (max_length - cur_len))
                textlength.append([cur_len])

        textid = torch.LongTensor(textid)
        textmask = torch.LongTensor(textmask)
        textlength = torch.LongTensor(textlength)
        return raw_text, textid, textmask, textlength
    def split_entity_input(self,label_sequence):
        entity_mark = dict()
        entity_pointer = None
        #print(label_sequence)
        for index, label in enumerate(label_sequence):
            if label.startswith('B'):
                category = label.split('_')[1]
                entity_pointer = (index, category)
                entity_mark.setdefault(entity_pointer, [label])
            elif label.startswith('S'):
                category = label.split('_')[1]
                entity_pointer = (index, category)
                entity_mark.setdefault(entity_pointer, [label])
                entity_pointer = None
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
    def predict_sentence(self, sentence):
        tag_dic = {"dis": "疾病", "bod": "身体"}
        if sentence == '':
            print("输入为空！请重新输入")
            return
        if len(sentence) > 198:
            print("输入句子过长，请输入小于198的长度字符！")
            sentence = sentence[:198]
        raw_text, test_ids, test_masks, test_lengths = self.from_input(sentence)
        test_dataset = TensorDataset(test_ids, test_masks, test_lengths)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)
        self.model.load_state_dict(torch.load(self.NEWPATH, map_location=device))
        self.model.eval()

        for i, dev_batch in enumerate(test_loader):
            sentence, masks, lengths = dev_batch
            batch_raw_text = raw_text[i]
            sentence, masks, lengths = Variable(sentence), Variable(masks), Variable(lengths)
            if use_cuda:
                sentence = sentence.to(device)
                masks = masks.to(device)

            predict_tags = self.model(sentence, masks)
            predict_tags.tolist()
            predict_tags = [i2l_dic[t.item()] for t in predict_tags[0]]
            predict_tags = predict_tags[:len(batch_raw_text)]
            pred = predict_tags[1:-1]
            raw_text = batch_raw_text[1:-1]
            entity_mark = self.split_entity_input(pred)
            entity_list = {}
            if entity_mark is not None:
                for item, ent in entity_mark.items():
                    # print(item, ent)
                    entity = ''
                    index, tag = item[0], item[1]
                    len_entity = len(ent)

                    for i in range(index, index + len_entity):
                        entity = entity + raw_text[i]
                    entity_list[tag_dic[tag]] = entity
            #print(entity_list)
        return entity_list
    def predict_file(self, input_file, output_file):
        tag_dic = {"dis": "疾病", "bod": "身体"}
        raw_text, test_ids, test_masks, test_lengths = self.from_txt(input_file)
        test_dataset = TensorDataset(test_ids, test_masks, test_lengths)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)
        self.model.load_state_dict(torch.load(self.NEWPATH, map_location=device))
        self.model.eval()
        op_file = codecs.open(output_file, 'w', 'utf-8')
        
        for i, dev_batch in enumerate(test_loader):
            sentence, masks, lengths = dev_batch
            batch_raw_text = raw_text[i]
            sentence, masks, lengths = Variable(sentence), Variable(masks), Variable(lengths)
            if use_cuda:
                sentence = sentence.to(device)
                masks = masks.to(device)

            predict_tags = self.model(sentence, masks)
            predict_tags.tolist()
            predict_tags = self.model(sentence, masks)
            predict_tags.tolist()
            predict_tags = [i2l_dic[t.item()] for t in predict_tags[0]]
            predict_tags = predict_tags[:len(batch_raw_text)]
            pred = predict_tags[1:-1]
           # print(pred)
            raw_text = batch_raw_text[1:-1]
            

            entity_mark = self.split_entity_input(pred)
            entity_list = {}
            if entity_mark is not None:
                for item, ent in entity_mark.items():
                    entity = ''
                    index, tag = item[0], item[1]
                    len_entity = len(ent)
                    for i in range(index, index + len_entity):
                        entity = entity + raw_text[i]
                    entity_list[tag_dic[tag]] = entity
            op_file.write("".join(raw_text))
            op_file.write("\n")
            op_file.write(json.dumps(entity_list, ensure_ascii=False))
            op_file.write("\n")

        op_file.close()
        print('处理完成！')
        print("结果保存至 {}".format(output_file))


