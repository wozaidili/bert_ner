# -*- coding: utf-8 -*-
import torch
# tag-entity:{dis:疾病 sym:临床表现 bod:身体 equ:医疗设备 pro:医疗程序 mic:微生物类 dep:科室 ite:医学检验项目 dru:药物}
l2i_dic = {"O": 0,
           "B_dis": 1,  "M_dis": 2,  "E_dis": 3,  "B_bod": 4,  "M_bod": 5, "E_bod": 6,  "<pad>": 7,"<start>": 8, "<eos>": 9}

i2l_dic ={0: 'O', 1: 'B_dis', 2: 'M_dis', 3: 'E_dis',  4: 'B_bod', 5: 'M_bod', 6: 'E_bod', 7: '<pad>', 8: '<start>', 9: '<eos>'}

train_file = 'data/train.txt'
dev_file = 'data/dev.txt'
test_file = 'data/test.txt'
vocab_file = 'data/chinese_wwm_pytorch/vocab.txt'


save_model_dir =  'data/model/'
medical_tool_model = 'data/model/params.pkl'
max_length = 100
batch_size = 1
epochs = 5
tagset_size = len(l2i_dic)
use_cuda = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
