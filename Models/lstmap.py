# -*- coding: utf-8 -*-
"""

#Bibliotecas e Sementes
"""

#Plotagem
import plotly.graph_objects as go
import matplotlib.pyplot as plt


#Dados
import numpy as np
import random as python_random
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

#IA
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import GRU

from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

#Mat
import math as m
import math
from sklearn.metrics import mean_squared_error

def reset_seeds():
   np.random.seed(4) 
   python_random.seed(9498)
   tf.random.set_seed(9)

reset_seeds()

"""#Dados"""

yolov3_320_1 = pd.read_excel('/content/drive/MyDrive/Dados_Finais/Dados/Yolov3-320/Dados1/Dados.xlsx')['TEMP']
yolov3_320_2 = pd.read_excel('/content/drive/MyDrive/Dados_Finais/Dados/Yolov3-320/Dados2/Dados.xlsx')['TEMP']
yolov4_tiny_320_1 = pd.read_excel('/content/drive/MyDrive/Dados/Yolov4-tiny-320/Dados1/Dados.xlsx')["TEMP"]
yolov4_tiny_320_2 = pd.read_excel('/content/drive/MyDrive/Dados/Yolov4-tiny-320/Dados2/Dados.xlsx')["TEMP"]
yolov4_tiny_416_1 = pd.read_excel('/content/drive/MyDrive/Dados/Yolov4-tiny-416/Dados1/Dados.xlsx')["TEMP"]
yolov4_tiny_416_2 = pd.read_excel('/content/drive/MyDrive/Dados/Yolov4-tiny-416/Dados2/Dados.xlsx')["TEMP"]
yolov4_tiny_512_1 = pd.read_excel('/content/drive/MyDrive/Dados/Yolov4-tiny-512/Dados1/Dados.xlsx')["TEMP"]
yolov4_tiny_512_2 = pd.read_excel('/content/drive/MyDrive/Dados/Yolov4-tiny-512/Dados2/Dados.xlsx')["TEMP"]
yolov4_320_1 = pd.read_excel('/content/drive/MyDrive/Dados/Yolov4-320/Dados1/Dados.xlsx')["TEMP"]
yolov4_320_2 = pd.read_excel('/content/drive/MyDrive/Dados/Yolov4-320/Dados2/Dados.xlsx')["TEMP"]
yolov4_416_1 = pd.read_excel('/content/drive/MyDrive/Dados/Yolov4-416/Dados1/Dados.xlsx')["TEMP"]
yolov4_416_2 = pd.read_excel('/content/drive/MyDrive/Dados/Yolov4-416/Dados2/Dados.xlsx')["TEMP"]
yolov4_512_1 = pd.read_excel('/content/drive/MyDrive/Dados/Yolov4-512/Dados1/Dados.xlsx')["TEMP"]
yolov4_512_2 = pd.read_excel('/content/drive/MyDrive/Dados/Yolov4-512/Dados2/Dados.xlsx')["TEMP"]
Mask_1 = pd.read_excel('/content/drive/MyDrive/Dados_Finais/Dados/mask-r-cnn/Dados1/Dados.xlsx')["TEMP"]
Mask_2 = pd.read_excel('/content/drive/MyDrive/Dados_Finais/Dados/mask-r-cnn/Dados2/Dados.xlsx')["TEMP"]

Dados=np.append(np.array(yolov3_320_1),[np.array(yolov3_320_2),np.array(yolov4_tiny_320_1),np.array(yolov4_tiny_320_2),np.array(yolov4_tiny_416_1),
                                        np.array(yolov4_tiny_416_2),np.array(yolov4_tiny_512_1),np.array(yolov4_tiny_512_2),np.array(yolov4_320_1),
                                        np.array(yolov4_320_2),np.array(yolov4_416_1),np.array(yolov4_416_2),np.array(yolov4_512_1),np.array(yolov4_512_2),
                                        np.array(Mask_1),np.array(Mask_2)])
print(len(Dados))

x=np.array(Dados)
#print(x)
#print(type(x))

x=x.astype('float32')
#print(x)
print(type(x))

scaler = MinMaxScaler(feature_range=(-1, 1))
x = scaler.fit_transform(x.reshape(-1,1))
print(x[0:5])
print(type(x))

look_back = 30
future = 15

X_train = []
y_train = []
for i in range(look_back, len(x)-future):
    X_train.append(x[i-look_back:i, 0])
    y_train.append(x[i:i+future, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

print(X_train[0],y_train[0])

input_shape = (X_train.shape[1], 1)
input_shape

"""#Modelo"""

regressor = Sequential()
regressor.add(LSTM(units =64, return_sequences = False, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.3))
#regressor.add(LSTM(units = 64, return_sequences = True))
#regressor.add(LSTM(units = 32, return_sequences = True))
#regressor.add(LSTM(units = 32))
regressor.add(Dense(units = 15))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
history = regressor.fit(X_train, y_train, epochs = 100, batch_size = 4, validation_split=0.2,callbacks=[early_stop])

inp_=pd.read_excel('/content/drive/MyDrive/Dados_Finais/Dados/Yolov3-320/Dados3/Dados.xlsx')['TEMP'][440:500]
inp_=list(inp_)

lista=pd.read_excel('/content/drive/MyDrive/Dados_Finais/Dados/Yolov3-320/Dados3/Dados.xlsx')
lista1=pd.read_excel('/content/drive/MyDrive/Dados_Finais/Dados/Yolov3-416/Dados3/Dados.xlsx')
lista2=pd.read_excel('/content/drive/MyDrive/Dados/Yolov4-tiny-320/Dados3/Dados.xlsx')
lista3=pd.read_excel('/content/drive/MyDrive/Dados/Yolov4-tiny-416/Dados3/Dados.xlsx')
lista4=pd.read_excel('/content/drive/MyDrive/Dados/Yolov4-tiny-512/Dados3/Dados.xlsx')
lista5=pd.read_excel('/content/drive/MyDrive/Dados/Yolov4-320/Dados3/Dados.xlsx')
lista6=pd.read_excel('/content/drive/MyDrive/Dados/Yolov4-416/Dados3/Dados.xlsx')
lista7=pd.read_excel('/content/drive/MyDrive/Dados/Yolov4-512/Dados3/Dados.xlsx')
lista8=pd.read_excel('/content/drive/MyDrive/Dados_Finais/Dados/mask-r-cnn/Dados3/Dados.xlsx')
Lista=[lista['TEMP'][60:75],
       lista['TEMP'][175:190],
       lista['TEMP'][290:305],
       lista['TEMP'][405:420],
       lista['TEMP'][520:535],
       lista['TEMP'][635:650],
       lista['TEMP'][750:765],
       lista['TEMP'][865:880],
       lista['TEMP'][980:995],
       lista['TEMP'][995:1010],
       lista['TEMP'][1110:1125],
       lista['TEMP'][1225:1240],
       lista['TEMP'][1340:1355],
       lista['TEMP'][1455:1470],
       lista['TEMP'][1570:1585],
       lista['TEMP'][1685:1700],
       lista1['TEMP'][60:75],
       lista1['TEMP'][175:190],
       lista1['TEMP'][290:305],
       lista1['TEMP'][405:420],
       lista1['TEMP'][520:535],
       lista1['TEMP'][635:650],
       lista1['TEMP'][750:765],
       lista1['TEMP'][865:880],
       lista1['TEMP'][980:995],
       lista1['TEMP'][995:1010],
       lista1['TEMP'][1110:1125],
       lista1['TEMP'][1225:1240],
       lista1['TEMP'][1340:1355],
       lista1['TEMP'][1455:1470],
       lista1['TEMP'][1570:1585],
       lista1['TEMP'][1685:1700],
       lista2['TEMP'][60:75],
       lista2['TEMP'][175:190],
       lista2['TEMP'][290:305],
       lista2['TEMP'][405:420],
       lista2['TEMP'][520:535],
       lista2['TEMP'][635:650],
       lista2['TEMP'][750:765],
       lista2['TEMP'][865:880],
       lista2['TEMP'][980:995],
       lista2['TEMP'][995:1010],
       lista2['TEMP'][1110:1125],
       lista2['TEMP'][1225:1240],
       lista2['TEMP'][1340:1355],
       lista2['TEMP'][1455:1470],
       lista2['TEMP'][1570:1585],
       lista2['TEMP'][1685:1700],
       lista3['TEMP'][60:75],
       lista3['TEMP'][175:190],
       lista3['TEMP'][290:305],
       lista3['TEMP'][405:420],
       lista3['TEMP'][520:535],
       lista3['TEMP'][635:650],
       lista3['TEMP'][750:765],
       lista3['TEMP'][865:880],
       lista3['TEMP'][980:995],
       lista3['TEMP'][995:1010],
       lista3['TEMP'][1110:1125],
       lista3['TEMP'][1225:1240],
       lista3['TEMP'][1340:1355],
       lista3['TEMP'][1455:1470],
       lista3['TEMP'][1570:1585],
       lista3['TEMP'][1685:1700],
       lista4['TEMP'][60:75],
       lista4['TEMP'][175:190],
       lista4['TEMP'][290:305],
       lista4['TEMP'][405:420],
       lista4['TEMP'][520:535],
       lista4['TEMP'][635:650],
       lista4['TEMP'][750:765],
       lista4['TEMP'][865:880],
       lista4['TEMP'][980:995],
       lista4['TEMP'][995:1010],
       lista4['TEMP'][1110:1125],
       lista4['TEMP'][1225:1240],
       lista4['TEMP'][1340:1355],
       lista4['TEMP'][1455:1470],
       lista4['TEMP'][1570:1585],
       lista4['TEMP'][1685:1700],
       lista5['TEMP'][60:75],
       lista5['TEMP'][175:190],
       lista5['TEMP'][290:305],
       lista5['TEMP'][405:420],
       lista5['TEMP'][520:535],
       lista5['TEMP'][635:650],
       lista5['TEMP'][750:765],
       lista5['TEMP'][865:880],
       lista5['TEMP'][980:995],
       lista5['TEMP'][995:1010],
       lista5['TEMP'][1110:1125],
       lista5['TEMP'][1225:1240],
       lista5['TEMP'][1340:1355],
       lista5['TEMP'][1455:1470],
       lista5['TEMP'][1570:1585],
       lista5['TEMP'][1685:1700],
       lista6['TEMP'][60:75],
       lista6['TEMP'][175:190],
       lista6['TEMP'][290:305],
       lista6['TEMP'][405:420],
       lista6['TEMP'][520:535],
       lista6['TEMP'][635:650],
       lista6['TEMP'][750:765],
       lista6['TEMP'][865:880],
       lista6['TEMP'][980:995],
       lista6['TEMP'][995:1010],
       lista6['TEMP'][1110:1125],
       lista6['TEMP'][1225:1240],
       lista6['TEMP'][1340:1355],
       lista6['TEMP'][1455:1470],
       lista6['TEMP'][1570:1585],
       lista6['TEMP'][1685:1700],
       lista7['TEMP'][60:75],
       lista7['TEMP'][175:190],
       lista7['TEMP'][290:305],
       lista7['TEMP'][405:420],
       lista7['TEMP'][520:535],
       lista7['TEMP'][635:650],
       lista7['TEMP'][750:765],
       lista7['TEMP'][865:880],
       lista7['TEMP'][980:995],
       lista7['TEMP'][995:1010],
       lista7['TEMP'][1110:1125],
       lista7['TEMP'][1225:1240],
       lista7['TEMP'][1340:1355],
       lista7['TEMP'][1455:1470],
       lista7['TEMP'][1570:1585],
       lista7['TEMP'][1685:1700],
       lista8['TEMP'][60:75],
       lista8['TEMP'][175:190],
       lista8['TEMP'][290:305],
       lista8['TEMP'][405:420],
       lista8['TEMP'][520:535],
       lista8['TEMP'][635:650],
       lista8['TEMP'][750:765],
       lista8['TEMP'][865:880],
       lista8['TEMP'][980:995],
       lista8['TEMP'][995:1010],
       lista8['TEMP'][1110:1125],
       lista8['TEMP'][1225:1240],
       lista8['TEMP'][1340:1355],
       lista8['TEMP'][1455:1470],
       lista8['TEMP'][1570:1585],
       lista8['TEMP'][1685:1700]]
Lista_total=[]
for i in Lista:
  Lista_total.append(i)

Lista_total=np.array(Lista_total)
Lista_total=np.reshape(Lista_total, 2160)
print(Lista_total)

inp_15=[lista['TEMP'][30:60],
       lista['TEMP'][145:175],
       lista['TEMP'][260:290],
       lista['TEMP'][375:405],
       lista['TEMP'][490:520],
       lista['TEMP'][605:635],
       lista['TEMP'][720:750],
       lista['TEMP'][835:865],
       lista['TEMP'][950:980],
       lista['TEMP'][965:995],
       lista['TEMP'][1080:1110],
       lista['TEMP'][1195:1225],
       lista['TEMP'][1310:1340],
       lista['TEMP'][1425:1455],
       lista['TEMP'][1540:1570],
       lista['TEMP'][1655:1685],
       lista1['TEMP'][30:60],
       lista1['TEMP'][145:175],
       lista1['TEMP'][260:290],
       lista1['TEMP'][375:405],
       lista1['TEMP'][490:520],
       lista1['TEMP'][605:635],
       lista1['TEMP'][720:750],
       lista1['TEMP'][835:865],
       lista1['TEMP'][950:980],
       lista1['TEMP'][965:995],
       lista1['TEMP'][1080:1110],
       lista1['TEMP'][1195:1225],
       lista1['TEMP'][1310:1340],
       lista1['TEMP'][1425:1455],
       lista1['TEMP'][1540:1570],
       lista1['TEMP'][1655:1685],
       lista2['TEMP'][30:60],
       lista2['TEMP'][145:175],
       lista2['TEMP'][260:290],
       lista2['TEMP'][375:405],
       lista2['TEMP'][490:520],
       lista2['TEMP'][605:635],
       lista2['TEMP'][720:750],
       lista2['TEMP'][835:865],
       lista2['TEMP'][950:980],
       lista2['TEMP'][965:995],
       lista2['TEMP'][1080:1110],
       lista2['TEMP'][1195:1225],
       lista2['TEMP'][1310:1340],
       lista2['TEMP'][1425:1455],
       lista2['TEMP'][1540:1570],
       lista2['TEMP'][1655:1685],
       lista3['TEMP'][30:60],
       lista3['TEMP'][145:175],
       lista3['TEMP'][260:290],
       lista3['TEMP'][375:405],
       lista3['TEMP'][490:520],
       lista3['TEMP'][605:635],
       lista3['TEMP'][720:750],
       lista3['TEMP'][835:865],
       lista3['TEMP'][950:980],
       lista3['TEMP'][965:995],
       lista3['TEMP'][1080:1110],
       lista3['TEMP'][1195:1225],
       lista3['TEMP'][1310:1340],
       lista3['TEMP'][1425:1455],
       lista3['TEMP'][1540:1570],
       lista3['TEMP'][1655:1685],
       lista4['TEMP'][30:60],
       lista4['TEMP'][145:175],
       lista4['TEMP'][260:290],
       lista4['TEMP'][375:405],
       lista4['TEMP'][490:520],
       lista4['TEMP'][605:635],
       lista4['TEMP'][720:750],
       lista4['TEMP'][835:865],
       lista4['TEMP'][950:980],
       lista4['TEMP'][965:995],
       lista4['TEMP'][1080:1110],
       lista4['TEMP'][1195:1225],
       lista4['TEMP'][1310:1340],
       lista4['TEMP'][1425:1455],
       lista4['TEMP'][1540:1570],
       lista4['TEMP'][1655:1685],
       lista5['TEMP'][30:60],
       lista5['TEMP'][145:175],
       lista5['TEMP'][260:290],
       lista5['TEMP'][375:405],
       lista5['TEMP'][490:520],
       lista5['TEMP'][605:635],
       lista5['TEMP'][720:750],
       lista5['TEMP'][835:865],
       lista5['TEMP'][950:980],
       lista5['TEMP'][965:995],
       lista5['TEMP'][1080:1110],
       lista5['TEMP'][1195:1225],
       lista5['TEMP'][1310:1340],
       lista5['TEMP'][1425:1455],
       lista5['TEMP'][1540:1570],
       lista5['TEMP'][1655:1685],
       lista6['TEMP'][30:60],
       lista6['TEMP'][145:175],
       lista6['TEMP'][260:290],
       lista6['TEMP'][375:405],
       lista6['TEMP'][490:520],
       lista6['TEMP'][605:635],
       lista6['TEMP'][720:750],
       lista6['TEMP'][835:865],
       lista6['TEMP'][950:980],
       lista6['TEMP'][965:995],
       lista6['TEMP'][1080:1110],
       lista6['TEMP'][1195:1225],
       lista6['TEMP'][1310:1340],
       lista6['TEMP'][1425:1455],
       lista6['TEMP'][1540:1570],
       lista6['TEMP'][1655:1685],
       lista7['TEMP'][30:60],
       lista7['TEMP'][145:175],
       lista7['TEMP'][260:290],
       lista7['TEMP'][375:405],
       lista7['TEMP'][490:520],
       lista7['TEMP'][605:635],
       lista7['TEMP'][720:750],
       lista7['TEMP'][835:865],
       lista7['TEMP'][950:980],
       lista7['TEMP'][965:995],
       lista7['TEMP'][1080:1110],
       lista7['TEMP'][1195:1225],
       lista7['TEMP'][1310:1340],
       lista7['TEMP'][1425:1455],
       lista7['TEMP'][1540:1570],
       lista7['TEMP'][1655:1685],
       lista8['TEMP'][30:60],
       lista8['TEMP'][145:175],
       lista8['TEMP'][260:290],
       lista8['TEMP'][375:405],
       lista8['TEMP'][490:520],
       lista8['TEMP'][605:635],
       lista8['TEMP'][720:750],
       lista8['TEMP'][835:865],
       lista8['TEMP'][950:980],
       lista8['TEMP'][965:995],
       lista8['TEMP'][1080:1110],
       lista8['TEMP'][1195:1225],
       lista8['TEMP'][1310:1340],
       lista8['TEMP'][1425:1455],
       lista8['TEMP'][1540:1570],
       lista8['TEMP'][1655:1685],]

prev=[]
for i in inp_15:
  inp_=i
  inp_=list(inp_)
  test=[]
  previsões=[]
  inp=np.array(inp_)
  inp=inp.reshape(-1,1)
  inp=scaler.transform(inp)
  test.append(inp)
  test=np.array(test)
  test=np.reshape(test, (test.shape[0], test.shape[1], 1))
  predicted = regressor.predict([test])
  predicted= scaler.inverse_transform(predicted)
  predicted=predicted[0]
  prev.append(predicted)

prev=np.reshape(prev, 2160)

plt.figure(figsize=(15,5))  
plt.plot(Lista_total, color='Black',label="Data",linestyle='--')
plt.plot(prev, color='red',label="Predict")
plt.title('Predicted temperature ', fontsize=14)
plt.ylabel('Temperature °C', fontsize=14)
plt.legend()
plt.grid(True)
plt.savefig('ResultadosMLP_3.png',dpi=300)
plt.show()

erro=mean_squared_error(prev,Lista_total)
print(erro)

max=[]
for i in range(2160):
  max.append(prev[i]+(erro/2))

a=0
b=0
c=0
d=0
for i in range(2160):
  if Lista_total[i] >70 and prev[i] < 70:
    a=a+1
  if Lista_total[i] >70 and prev[i] > 70:
    b=b+1
  if Lista_total[i] <70 and prev[i] > 70:
    c=c+1
    #print(i,Lista_total[i],prev[i])
  if Lista_total[i] <70 and prev[i] < 70:
    d=d+1

acc=(b)/(a+b+c)
print(a)
print(b)
print(c)
print(acc)

a=0
b=0
c=0
d=0
for i in range(2157):
  if Lista_total[i] >70 and Lista_total[i+1] >70 and Lista_total[i+2] >70 and prev[i] < 70:
    a=a+1
    #print(Lista_total[i],Lista_total[i+1],Lista_total[i+2],prev[i])
  if Lista_total[i] >70 and Lista_total[i+1] >70 and Lista_total[i+2] >70 and prev[i] > 70:
    b=b+1
    #print(Lista_total[i],Lista_total[i+1],Lista_total[i+2],prev[i])
  if Lista_total[i] <70 and Lista_total[i+1] <70 and Lista_total[i+2] <70 and prev[i] > 70:
    c=c+1
    #print(i,Lista_total[i],prev[i])
  if Lista_total[i] <70 and prev[i] < 70:
    d=d+1

acc=(b)/(a+b+c)
print(a)
print(b)
print(c)
print(acc)

a=0
b=0
c=0

for i in range(2155):
  if Lista_total[i] >70 and Lista_total[i+1] >70 and Lista_total[i+2] >70 and Lista_total[i+3] >70 and Lista_total[i+4] >70  and prev[i] < 70:
    a=a+1
    #print(Lista_total[i],Lista_total[i+1],Lista_total[i+2],prev[i])
  if Lista_total[i] >70 and Lista_total[i+1] >70 and Lista_total[i+2] >70 and Lista_total[i+3] >70 and Lista_total[i+4] >70 and prev[i] > 70:
    b=b+1
    #print(Lista_total[i],Lista_total[i+1],Lista_total[i+2],prev[i])
  if Lista_total[i] <70 and Lista_total[i+1] <70 and Lista_total[i+2] <70 and Lista_total[i+3] <70 and Lista_total[i+4] <70 and prev[i] > 70:
    c=c+1

acc=(b)/(a+b+c)
print(a)
print(b)
print(c)
print(acc)

if 1 >1.2 and 2 > 1.2:
  print(1)

regressor.save("/content/drive/MyDrive/lstm.h5")

import keras
regressor =  keras.models.load_model("/content/drive/MyDrive/lstm.h5")