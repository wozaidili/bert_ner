**使用方法参见predict.py**

### 医学实体识别工具
依赖的库 python3, pytorch, json, numpy, torch, tqdm
调整的参数和模型在constant.py中
数据集：训练集、验证集、测试集：
	train_file = './data/train.txt'
	dev_file = './data/val.txt'
	test_file = './data/test.txt'
测试文件和预测使用的模型地址为：
	medical_tool_model = './data/model/model.pkl'



### 训练
直接运行该命令进行训练模型
python train.py

### 测试
直接运行该命令进行测试函数：返回一个图表数据，显示不同实体类别上的准确率、召回率、F1值，以及总体上的各指标值。
python test_file.py

### 测试结果

+--------------+--------+--------+-------+
|   实体类别   | 准确率 | 召回率 |  F1值 |
+--------------+--------+--------+-------+
|     疾病     | 0.794  | 0.823  | 0.808 |
|   临床表现   | 0.569  | 0.612  |  0.59 |
|     身体     | 0.682  | 0.715  | 0.698 |
|   医疗程序   | 0.669  | 0.701  | 0.685 |
|   医疗设备   | 0.707  | 0.719  | 0.713 |
|     药物     | 0.833  | 0.857  | 0.845 |
| 医学检验项目 | 0.615  | 0.595  | 0.605 |
|     科室     | 0.611  | 0.667  | 0.638 |
|   微生物类   | 0.764  | 0.713  | 0.738 |
|     总体     | 0.696  | 0.725  |  0.71 |
### 训练
修改 constants.py 和 train.py 中的tag

bert 与训练模型 data/chinese_wwm_pytorch
训练权重保存模型 data/model

训练：python train.py

### 预测

导入方法：from run import medical_ner
新建实例：my_pred = medical_ner()

medical_ner类提供两个接口预测函数

1 predict_sentence(sentence)
arg:sentence 单个字符串句子
测试单个句子，返回:{"实体类别"：“实体”},不同实体以逗号隔开


2 predict_file(input_file, output_file)
arg:input_file:输入文件名 output_file:输出文件名
测试整个文件
文件格式每行待提取实体的句子和提取出的实体{"实体类别"：“实体”},不同实体以逗号隔开
