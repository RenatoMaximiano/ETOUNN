# -*- coding: utf-8 -*-
"""

#MLP para estimativa do consumo de energia da raspberry pi 4 rodando diferentes algoritmos de visão computacional para detecção de objetos

#Importando Bibliotecas e definindo sementes
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import random as python_random
import tensorflow as tf

# % matplotlib inline

def reset_seeds():
   np.random.seed(4) 
   python_random.seed(9498)
   tf.random.set_seed(9)

reset_seeds()

"""#Importando Dados """

df = pd.read_excel('drive/MyDrive/testecorr.xlsx')
print(len(df))

#df.columns = ['TEMP', 'CPU', 'MEM', 'CURR']

df

"""#Vizualizando correlação de variaveis"""

plt.figure(figsize=(6,5))
sns.heatmap(df.corr(), annot=True)
plt.show()

"""#Divisão de dados para treinamento e teste"""

X = df.drop('Corr', axis = 1) 
X = X.drop('MEM', axis = 1) 
y = df ['Corr']
X=X.astype('float32')
y=y.astype('float32')
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.2, random_state = 9)

X_train

"""#Normalizar entre -1 e 1"""

from sklearn.preprocessing import MinMaxScaler
#print(X_test)
scaler = MinMaxScaler(feature_range=(-1, 1)) 
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test)

"""#Modelo"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
model = Sequential()
n= 256 
n2 = 256 
n3 =  128
n4 =  256 
n5=512
model.add(Dense(n,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(n2,activation='relu'))
model.add(Dense(n3,activation='relu'))
model.add(Dense(n4,activation='relu'))
model.add(Dense(n5,activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

"""#Realizanção do treinamento"""

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
model.fit(x=X_train,y=y_train.values,
          validation_split=0.2,
          batch_size=12,epochs=90, callbacks=[early_stop])

plt.figure(figsize=(15,5))    
losses = pd.DataFrame(model.history.history)
losses.plot()

from keras.utils.vis_utils import plot_model

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

pip install keras-visualizer

from keras_visualizer import visualizer 
visualizer(model,format='png', view=True)

from sklearn.metrics import mean_squared_error,mean_absolute_error 
from sklearn.metrics import r2_score
predictions = model.predict(X_test)
mean_absolute_error(y_test,predictions)

mean_squared_error(y_test,predictions)

r2_score(y_test,predictions)

p=np.reshape(predictions, len(predictions))

print( len(p))
print( len(predictions))
mean_absolute_error(y_test,p)

y=np.array(y_test)

y=np.reshape(y_test, len(y_test))

import plotly.graph_objects as go
import matplotlib.pyplot as plt
#p,y_test = zip(*sorted(zip(p,y_test)))
fig = go.Figure()
fig.add_trace(go.Scatter(y=p,name="previsão"))#,mode='markers',marker_symbol="x"))
fig.add_trace(go.Scatter(y=y_test,name="Dados"))#,mode='markers'))
fig.update_layout(title='Dados simples',
                   xaxis_title='Unidade',
                   yaxis_title='corrente')
fig.show()

import matplotlib.pyplot as plt
plt.figure(figsize=(15,5))    

plt.plot([*range(0, 116, 1)],y_test[0:116], color='Black',label="Data",linestyle='--',marker='o')
plt.plot(p[0:116], color='red',label="Predict",marker='X')
plt.title('Current estimate', fontsize=14)
plt.ylabel('A', fontsize=14)
plt.legend()
plt.grid(True)
plt.savefig('ResultadosMLP_0.png',dpi=300)
plt.show()

plt.figure(figsize=(15,5))  
plt.plot([*range(0, 116, 1)],y_test[116:232], color='Black',label="Data",linestyle='--',marker='o')
plt.plot(p[116:232], color='red',label="Predict",marker='X')
plt.title('Current estimate', fontsize=14)
plt.ylabel('A', fontsize=14)
plt.legend()
plt.grid(True)
plt.savefig('ResultadosMLP_1.png',dpi=300)
plt.show()

plt.figure(figsize=(15,5))  
plt.plot([*range(0, 116, 1)],y_test[232:348], color='Black',label="Data",linestyle='--',marker='o')
plt.plot(p[232:348], color='red',label="Predict",marker='X')
plt.title('Current estimate', fontsize=14)
plt.ylabel('A', fontsize=14)
plt.legend()
plt.grid(True)
plt.savefig('ResultadosMLP_2.png',dpi=300)
plt.show()

plt.figure(figsize=(15,5))  
plt.plot([*range(0, 116, 1)],y_test[348:464], color='Black',label="Data",linestyle='--',marker='o')
plt.plot(p[348:464], color='red',label="Predict",marker='X')
plt.title('Current estimate', fontsize=14)
plt.ylabel('A', fontsize=14)
plt.legend()
plt.grid(True)
plt.savefig('ResultadosMLP_3.png',dpi=300)
plt.show()

plt.figure(figsize=(15,5))  
plt.plot([*range(0, 465, 1)],y_test, color='Black',label="Data",linestyle='--',marker='o')
plt.plot(p, color='red',label="Predict",marker='X',linestyle='-')
plt.ylabel('Current (A)', fontsize=16)
plt.xlabel('Actual values and estimated values', fontsize=16)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.legend()
plt.grid(True)
plt.savefig('ResultadosMLP_5.png',dpi=300)
plt.show()

plt.figure(figsize=(15,5))  
plt.plot(p,y_test, color='Black',label="Data",marker='o',linestyle='')
plt.title('Current estimate', fontsize=14)
plt.ylabel('Current (A)', fontsize=14)
plt.legend()
plt.grid(True)
plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, p, 1))(np.unique(y_test)))
plt.savefig('ResultadosMLP_4.png',dpi=300)
plt.show()

from matplotlib.transforms import Bbox, TransformedBbox, blended_transform_factory

from mpl_toolkits.axes_grid1.inset_locator import BboxPatch, BboxConnector,\
    BboxConnectorPatch


def connect_bbox(bbox1, bbox2,
                 loc1a, loc2a, loc1b, loc2b,
                 prop_lines, prop_patches=None):
    if prop_patches is None:
        prop_patches = prop_lines.copy()
        prop_patches["alpha"] = prop_patches.get("alpha", 1)*0.2

    c1 = BboxConnector(bbox1, bbox2, loc1=loc1a, loc2=loc2a, **prop_lines)
    c1.set_clip_on(False)
    c2 = BboxConnector(bbox1, bbox2, loc1=loc1b, loc2=loc2b, **prop_lines)
    c2.set_clip_on(False)

    bbox_patch1 = BboxPatch(bbox1, **prop_patches)
    bbox_patch2 = BboxPatch(bbox2, **prop_patches)

    p = BboxConnectorPatch(bbox1, bbox2,
                           loc1a=loc1a, loc2a=loc2a, loc1b=loc1b, loc2b=loc2b,
                           **prop_patches)
    p.set_clip_on(False)

    return c1, c2, bbox_patch1, bbox_patch2, p




def zoom_effect02(ax1, ax2, **kwargs):
    """
    ax2 : the big main axes
    ax1 : the zoomed axes
    The xmin & xmax will be taken from the
    ax1.viewLim.
    """

    tt = ax1.transScale + (ax1.transLimits + ax2.transAxes)
    trans = blended_transform_factory(ax2.transData, tt)

    mybbox1 = ax1.bbox
    mybbox2 = TransformedBbox(ax1.viewLim, trans)

    prop_patches = kwargs.copy()
    prop_patches["ec"] = "none"
    prop_patches["alpha"] = 0.2

    c1, c2, bbox_patch1, bbox_patch2, p = \
        connect_bbox(mybbox1, mybbox2,
                     loc1a=2, loc2a=3, loc1b=1, loc2b=4, 
                     prop_lines=kwargs, prop_patches=prop_patches)

    ax1.add_patch(bbox_patch1)
    ax2.add_patch(bbox_patch2)
    ax2.add_patch(c1)
    ax2.add_patch(c2)
    ax2.add_patch(p)

    return c1, c2, bbox_patch1, bbox_patch2, p


import matplotlib.pyplot as plt
import numpy as np



plt.figure(figsize=(15,10))
ax1 = plt.subplot(2,1,1)
plt.plot([*range(0, 465, 1)],y_test, color='Black',label="Data",linestyle='--',marker='o')
plt.plot(p, color='red',label="Predict",marker='X',linestyle='-')
plt.ylabel('Current (A)', fontsize=16)
#plt.xlabel('Actual values and estimated values', fontsize=16)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.grid(True)

ax2 = plt.subplot(2,2,3)
plt.plot([*range(0, 24, 1)],y_test[0:24], color='Black',label="Data",linestyle='--',marker='o')
plt.plot(p[0:24], color='red',label="Predict",marker='X',linestyle='-')
plt.ylabel('Current (A)', fontsize=16)
#plt.xlabel('Actual values and estimated values', fontsize=16)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.grid(True)
zoom_effect02(ax2, ax1)
ax3 = plt.subplot(2,2,4)
plt.plot([*range(300, 324, 1)],y_test[300:324], color='Black',label="Data",linestyle='--',marker='o')
plt.plot([*range(300, 324, 1)],p[300:324], color='red',label="Predict",marker='X',linestyle='-')
plt.ylabel('Current (A)', fontsize=16)
#plt.xlabel('Actual values and estimated values', fontsize=16)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.grid(True)
zoom_effect02(ax3, ax1)
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -1.3),
          fancybox=True, shadow=True, ncol=2,fontsize=16)
plt.savefig('ResultadosMLP_6.png',dpi=300)
plt.show()

[*range(200, 224, 1)]

import statistics as s
s.mean(y_test)

s.mean(p)

"""#Salvando modelo e escala e aplicando teste"""

model.save("/content/drive/MyDrive/model2.h5")

import keras

model =  keras.models.load_model("/content/drive/MyDrive/model2.h5")

import joblib
scaler_filename = "/content/drive/MyDrive/scaler.txt"
joblib.dump(scaler, scaler_filename) 

# And now to load...

scaler = joblib.load(scaler_filename)

x=np.array([[40,	4]])
#x=x.reshape(-1,1)
x=scaler.transform(x) 
predictions = model.predict(x)
predictions

yolov4_tiny_320_1 = scaler.transform(pd.read_excel('/content/drive/MyDrive/Dados_Finais/Dados/mask-r-cnn/Dados3/Dados.xlsx'))
predictions0 = model.predict(yolov4_tiny_320_1)

import statistics as s
predictions0=np.reshape(predictions0, len(predictions0))
t=s.mean(predictions0)
print(t*5.15)