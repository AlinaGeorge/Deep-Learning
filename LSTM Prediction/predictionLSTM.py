import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Embedding

text="machine machine machine machine machine machine machine learning machine learning machine learning"

chars=sorted(set(text))
char_to_int={c:i for i,c in enumerate(chars)}
int_to_char={i:c for i,c in enumerate(chars)}

seq_length=4
X=[]
y=[]

for i in range(len(text)-seq_length):
    X.append([char_to_int[c] for c in text[i:i+seq_length]])
    y.append(char_to_int[text[i+seq_length]])

X=np.array(X)
y=np.array(y)

model=Sequential([
    Embedding(len(chars),16),
    LSTM(32),
    Dense(len(chars),activation="softmax")
])

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(X,y,epochs=30,verbose=0)

sequence=input("Enter sequence:")
x=np.array([[char_to_int[c] for c in sequence]])
prediction=model.predict(x,verbose=0)
result=int_to_char[np.argmax(prediction)]

print("Input:",sequence)
print("Predicted next character:",result)
