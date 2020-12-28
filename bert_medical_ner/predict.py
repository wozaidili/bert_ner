from run import medical_ner


#使用工具运行
my_pred=medical_ner()
#根据提示输入单句：“高血压病人不可食用阿莫西林等药物”
#sentence=input("输入需要测试的句子:")
#my_pred.predict_sentence("".join(sentence.split()))
sent="肾血管性高血压以单侧肾动脉狭窄为多发"
print("句子为：",sent)
print("预测实体为:",my_pred.predict_sentence(''.join(sent.split())))
#输入文件(测试文件，输出文件)
my_pred.predict_file("my_test.txt","outt.txt")
