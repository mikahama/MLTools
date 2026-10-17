from MLTools import seq2seq
import pandas as pd
from mikatools import open_read

#Load the data
source = open_read("train/train.vi").readlines()
target = open_read("train/train.zh").readlines()
data = list(zip(source, target))

#Split the data
train, validation, test = seq2seq.split(data)

#Train the model
model = seq2seq.train("translation_model", train, validation)

#Evaluate
model.evaluate(test)

#Print some example results
print(model.predict("Bạn có sống ở quanh đây không ?"))
print(model.predict("Tôi không_thể ăn được nữa ."))