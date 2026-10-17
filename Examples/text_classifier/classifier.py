from MLTools import seq2seq
import pandas as pd

#Load the CSV
df = pd.read_csv("sentiment_analysis.csv")
data = df[["text", "sentiment"]].values.tolist()

#Split the data
train, validation, test = seq2seq.split(data)

#Train the model
model = seq2seq.train("sentiment_model", train, validation)

#Evaluate
model.evaluate(test)

#Print some example results
print(model.predict("OMG WTF is this?"))
print(model.predict("I really like this cake"))