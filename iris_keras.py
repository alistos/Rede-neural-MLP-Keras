import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
from keras.optimizers import SGD
import pandas

#header=None para evitar que a primeira linha seja ignorada
dataframe = pandas.read_csv("iris.csv", delimiter=",", header=None)
dataset = dataframe.to_numpy()

X = dataset[:,0:4].astype(float)
y = dataset[:,4]

"""
    Métodos para fazer um "one-hot encoding" em classes que possuem valores categóricos
    prática recomendada ao lidar com classificadores que preferem trabalhar com valores
    númericos.
    Exemplo do que faz essa técnica:
    classes =  red, green, blue
    após o one-hot encoding, red = [1,0,0]; green = [0,1,0]; blue = [0,0,1]
"""

encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
bin_y = np_utils.to_categorical(encoded_Y)

X_train,X_test,y_train,y_test=train_test_split(X,bin_y,train_size=0.5,random_state=2)

#Sequential = modelo linear sequencial da rede neural, onde se pode adicionar layers separadamente. 
model=Sequential()

"""
    add = função para adicionar um novo layer; Dense = Layer densamente conectado; 16 = número de neurônios
    input_dim = númweo de inputs iniciais; activation = funções de ativação, EXs: sigmoid, relu, softmax
    ultimo layer usa função softmax para garantir que o valor de output esteja entre 0 e 1 e possa ser usado
    como probabilidade prevista.
"""
model.add(Dense(16, input_dim=4, activation="sigmoid"))
model.add(Dense(8, activation="sigmoid"))
model.add(Dense(3, activation="softmax"))

"""
    compile = função que configura o modelo para o treinamento
    optimizer = algoritmo de optimização, como o adam, SGD, etc., aqui uso o SGD (Gradient Descent (with momentum)
    SGD é interessante pois permite alterar a taxa de aprendizagem.
    loss = função de erro, categorical_crossentropy é a função default para problemas de classificação multiclasse, para binários
    seria binary_crossentropy, regressão seria mean_squared_error.
    metrics = lista de métricas a ser calculada durante a fase de treinamento e teste.
"""
opt = SGD(learning_rate=0.5)
model.compile(optimizer=opt,loss="categorical_crossentropy",metrics=["accuracy"])
#model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

#verbose = tipo de "print" que será mostrado com os resultados, possue os tipos 0, 1 e 2
model.fit(X_train,y_train,epochs=25, verbose=2)
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print("Accuracy = {:.2f}".format(accuracy))

