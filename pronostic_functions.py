#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
import seaborn as sns

FIG_SIZE = (18,10)
FONT_SIZE = 12
PERCENT_TRAIN = 0.70
SIZE_TITLE = 18


def obtener_estadisticos(X):
    def visualización_datos():
        ciclo, tend = sm.tsa.filters.hpfilter(X)
        plt.figure(figsize=(18, 4))
        plt.plot(X)
        plt.plot(tend)
        plt.title("Frecuencia de los servicios seleccionados", fontdict=None, size=18)
        plt.xticks(rotation=90, size=18);
    print(visualización_datos());

    def acf_plot(z):
        acf_X = acf(z, fft=False)
        plt.figure(figsize=(18, 4))
        plt.stem(range(len(acf_X)), acf_X, use_line_collection=True)
        plt.title("ACF Ciclo de la Función", fontdict=None, size=18)
        plt.grid()
    print(acf_plot(X));

    def pacf_plot(z):
        pacf_X = pacf(z)
        plt.figure(figsize=(18, 4))
        plt.stem(range(len(pacf_X)), pacf_X, use_line_collection=True)
        plt.title("PACF Tendencia de la Función", fontdict=None, size=18)
        plt.grid()
    print(pacf_plot(X));   

def obtener_datos(ciudad):
    df=pd.read_csv(ciudad+".csv",sep=';')#,parse_dates=['Fecha'],index_col=['Fecha'])
    df=df.dropna()
    obtener_estadisticos(df['servicios'])
    return df

def model_arima(df):
    p_value =  int(input('Ingrese valor p: '))
    #Separación datos de entrenamiento y test
    size = int(len(df)*PERCENT_TRAIN)
    df_train, df_test=df[0:size],df[size:len(df)]

    #Entrenamiento del modelo
    modelo = ARIMA(df_train["servicios"], order=(p_value, 0, 1))  
    resultados = modelo.fit()

    df_train['pronostico'] = resultados.fittedvalues  
    df_train=df_train.dropna()
    plt.figure(figsize=FIG_SIZE)
    plt.title("Ajuste Modelo ARIMA", fontdict=None, size=SIZE_TITLE)
    plt.plot(df_train['servicios'],'.-k')
    plt.plot(df_train['pronostico'],'-r')
    print("Mean absolute error: %.2f" % mean_absolute_error(df_train['servicios'], df_train['pronostico']))
    print("Mean squared error: %.2f" % mean_squared_error(df_train['servicios'], df_train['pronostico']))# MSE
    print("Variance score: %.2f" % r2_score(df_train['servicios'], df_train['pronostico'])) # R2
    
    #Predicción
    df_test['prediction'] = resultados.predict(len(df_train),len(df)-1)
    plt.figure(figsize=FIG_SIZE)
    plt.title("Predicción (TEST y Datos reales)", fontdict=None, size=SIZE_TITLE)
    plt.plot(df_test['servicios'],'.-k')
    plt.plot(df_test['prediction'],'.-r')
    
def model_adaline(df):
    ciclo, tend = sm.tsa.filters.hpfilter(df["servicios"])
    df['tend'] = tend
    #df[["servicios","tend"]].plot(figsize=(18, 5),fontsize=12)
    #legend = plt.legend()
    #legend.prop.set_size(14);
    data=df["servicios"]
    u = np.array(data[1:len(data)]) - np.array(data[0:len(data) - 1])
    class AdalineTS:
        def __init__(self,P=None,learning_rate=0.001):# número de retardos a usar, # tasa de aprendizaje
            self.P = P
            self.learning_rate = learning_rate
            self.X = []
            self.coef_ = [0.] * P
            self.intercept_ = 0.

        def predict(self):
            if len(self.X) < self.P:
                return None
            X = np.array(self.X)
            u = np.dot(X, self.coef_) + self.intercept_
            return u

        def fit(self, d):
            y = self.predict()
            if y is not None:
                e = d - y
                self.coef_ += 2 * self.learning_rate * e * np.array(self.X)
                self.intercept_ += 2 * self.learning_rate * e
            self.X.append(d)
            if len(self.X) > self.P:
                self.X.pop(0)
    optimal_learning_rate = None
    optimal_sse = None

    P = int(input("Ingrese el valor p: "))
    len_train=int(len(data)*0.80)
    for learning_rate in np.linspace(start=0.00000000001, stop=0.000001, num=100):
        adaline = AdalineTS(P=P,learning_rate=learning_rate)
        forecasts = []
        for z in u[0:len_train]:
            forecasts.append(adaline.predict())
            adaline.fit(z)
        sse = sum([(a-b)**2 for a,b in zip(u[P:len_train], forecasts[P:])])
        if optimal_sse is None or sse < optimal_sse:
            optimal_sse = sse
            optimal_learning_rate = learning_rate
    adaline = AdalineTS(P=P,learning_rate=optimal_learning_rate)
    forecasts = []
    for z in u:
        forecasts.append(adaline.predict())
        adaline.fit(z)
    s = [m + n for m,n in zip(data[P:len(data)-1], forecasts[P:])]
    #s = data[0:P+1] + s
    plt.figure(figsize=(20,6))
    plt.plot(data[P:-1],'.-k', lw=1, ms=5,alpha=1)
    #plt.plot(tend, '.-g', lw=1, ms=5, alpha=0.9)
    plt.plot(s, '.-r', lw=1, ms=5, alpha=0.8)

    plt.vlines([int(len(data)*0.80)], ls='--', color='blue', ymin=min(s), ymax=max(s))
    plt.show()

def modelo_perceptron_multicapa(df):
    df_train = df.servicios[0:299]
    df_test = df.servicios[299:]

    scaler = MinMaxScaler() # crea el transformador
    df_train_scaled = scaler.fit_transform(np.array(df_train).reshape(-1, 1)) # escala la serie  # z es un array de listas como efecto
    df_train_scaled = [u[0] for u in df_train_scaled]# z es un array de listas como efecto

    data_d1 = [df_train_scaled[t] - df_train_scaled[t - 1] for t in range(1, len(df_train_scaled))] # Se remueve la tendencia

    data_d1d12 = [data_d1[t] - data_d1[t - 7] for t in range(7, len(data_d1))] # Para remover la componente cíclica cada 8 días

    #Se contruye una matriz de regresores, Ya que la implementación dispobible en sklearn es para modelos de regresión, se debe armar una matrix donde las variables independientes son zt−1, …, zt−P y la variable dependiente es zt.

    P = int(input("Ingrese el valor p: "))
    X_train_regr = []

    for t in range(P - 1, len(data_d1d12) - 1):
        X_train_regr.append([data_d1d12[t - n] for n in range(P)])

    d = data_d1d12[P:]

    #Modelo de percdeptron multicalpa

    H = 4 # Se escoge arbitrariamente el numero de neuronas
    np.random.seed(123456)

    mlp = MLPRegressor(
        hidden_layer_sizes=(H, ),
        #activation = 'logistic',
        activation ='logistic',
        learning_rate = 'adaptive',
        momentum = 0.0,
        learning_rate_init = 0.002,
        max_iter = 100000)

    mlp.fit(X_train_regr, d)   # Entrenamiento

    y_d1d12_m2 = mlp.predict(X_train_regr)  # Pronostico

    #  transformaciones inversas a las realizadas
    y_d1d12_m2 = data_d1d12[0:P] + y_d1d12_m2.tolist()

    y_d1_m2 = [y_d1d12_m2[t] + data_d1[t] for t in range(len(y_d1d12_m2))] # inversa de la remoción del ciclo 
    y_d1_m2 = data_d1[0:7] + y_d1_m2

    y_m2 = [y_d1_m2[t] + df_train_scaled[t] for t in range(len(y_d1_m2))] # agregamos la tendencia 
    y_m2 = [df_train_scaled[0]] + y_m2

    y_m2 = scaler.inverse_transform([[u] for u in y_m2]) # Desescalamos los datos 
    y_m2 = [u[0] for u in y_m2.tolist()] 

    print("Mean absolute error: %.2f" % mean_absolute_error(df_train, y_m2))
    #print("Mean_absolute_error: %.2f" % mean_absolute_error(df_test, mlp.predict(df_test))

    print("Mean squared error: %.2f" % mean_squared_error(df_train, y_m2))# MSE
    #print("Mean squared error: %.2f" % mean_squared_error(df_test, y_m2))# MSE

    print("Variance score: %.2f" % r2_score(df_train, y_m2)) # R2
    #print("Variance score: %.2f" % r2_score(df_test, y_m2)) # R2

    plt.figure(figsize=(14, 5))
    plt.plot(df_train, ".-k")
    plt.plot(y_m2, "-r")
    plt.title("Ajuste del Modelo Perceptrón Multicapa", fontdict=None, size=18)
    plt.grid()


def modelo_perceptron_multicapa_sin_transformaciones(df):
    P = int(input('Ingrese valor p: '))

    def computar_modelo():
        n = np.random.rand(len(df)) < 0.8 
        df_train = df.servicios[n]
        df_test = df.servicios[~n]

        # crea el transformador
        scaler = MinMaxScaler()

        # escala la serie
        df_scaled = scaler.fit_transform(np.array(df.servicios).reshape(-1, 1))

        # z es un array de listas como efecto
        # del escalamiento
        df_scaled = [u[0] for u in df_scaled]

        #Se contruye una matriz de regresores
        #P = int(input("Ingrese valor p: "))
        X = []

        for t in range(P - 1, len(df) - 1):
            X.append([df_scaled[t - n] for n in range(P)])
                    
        observed_scaled = df_scaled[P:]

        #Modelo de percdeptron multicalpa

        np.random.seed(123456)

        H = 3  # Se escoge arbitrariamente

        mlp = MLPRegressor(
            hidden_layer_sizes=(H,),
            activation='tanh',
            learning_rate="adaptive",
            momentum=0.0,
            learning_rate_init=0.1,
            max_iter=10000,
        )

        # Entrenamiento
        mlp.fit(X[0:(len(X) - len(df_test))], observed_scaled[0:(len(X) - len(df_test))]) 

        # Pronostico
        y_scaled_m1 = mlp.predict(X)


        # Se desescala para llevar los valores a la escala de los datos originales
        y_m1 = scaler.inverse_transform([[u] for u in y_scaled_m1])
        y_m1 = [u[0] for u in y_m1]

        return y_m1

    def computar_metricas(): 

        y_m1 = computar_modelo()

        n = np.random.rand(len(df)) < 0.8 
        df_train = df.servicios[n]
        df_test = df.servicios[~n]

        #P = 8

        e_train = df_train[P:]
        e_y_m1_train = y_m1[:-len(df_test)]
        e_y_m1_test = y_m1[len(e_y_m1_train):]

        print("Mean absolute error: %.2f" % mean_absolute_error(e_train, e_y_m1_train)) 
        print("Mean_absolute_error: %.2f" % mean_absolute_error(e_train, e_y_m1_train))
        print("Mean squared error: %.2f" % mean_squared_error(e_train, e_y_m1_train))# MSE
        print("Mean squared error: %.2f" % mean_squared_error(e_train, e_y_m1_train))# MSE
        print("Variance score: %.2f" % r2_score(e_train, e_y_m1_train)) # R2
        print("Variance score: %.2f" % r2_score(e_train, e_y_m1_train)) # R2

    def graficar_modelo():

        y_m1 = computar_modelo()

        n = np.random.rand(len(df)) < 0.8 
        df_train = df.servicios[n]
        df_test = df.servicios[~n]

        #P = 8

        plt.figure(figsize=(14, 5))
        plt.plot(df.servicios, ".-k")
        plt.grid()
        plt.plot([None] * P + y_m1, "-r")
        plt.title("Ajuste del Modelo Perceptrón Multicapa", fontdict=None, size=18)
        plt.plot([len(df.servicios) - len(df_test), len(df.servicios) - len(df_test)], [min(df.servicios), max(df.servicios)], "--", linewidth=2)

    graficar_modelo()
    computar_metricas()