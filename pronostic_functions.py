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
import statsmodels
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
import seaborn as sns

FIG_SIZE = (14, 5)
FONT_SIZE = 12
PERCENT_TRAIN = 0.90
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
    #global df #Nuevo (Diego)

    df=pd.read_csv(ciudad+".csv",sep=';')#,parse_dates=['Fecha'],index_col=['Fecha'])
    df=df.dropna()
    obtener_estadisticos(df['servicios'])

    return df

def model_arima(df):
    
    
    p_value = int(input('Ingrese valor p para modelo ARIMA: '))
    #Separación datos de entrenamiento y test
    size = int(len(df)*PERCENT_TRAIN)
    df_train, df_test=df[0:size],df[size:len(df)]

    #Entrenamiento del modelo
    modelo = ARIMA(df_train["servicios"], order=(p_value, 0, 1))  
    resultados = modelo.fit()

    df['pronostico'] = resultados.predict(0,len(df)-1)
    test = resultados.predict(len(df_train),len(df)-1)
    
    df=df.dropna()
    plt.figure(figsize=FIG_SIZE)
    plt.title("Ajuste Modelo ARIMA", fontdict=None, size=SIZE_TITLE)
    plt.plot(df['servicios'],'.-k')
    plt.plot(df['pronostico'],'-r')
    plt.plot([len(df.servicios) - len(df_test), len(df.servicios) - len(df_test)], [min(df.servicios), max(df.servicios)], "--", linewidth=2)
    print("Mean absolute error: %.2f" % mean_absolute_error(df_test['servicios'], test))
    print("Mean absolute error: %.2f" % mean_absolute_error(df_train['servicios'],df['pronostico'][:size]))
    print("Mean squared error: %.2f" % mean_squared_error(df_test['servicios'], test))# MSE
    print("Mean squared error: %.2f" %mean_squared_error(df_train['servicios'], df['pronostico'][:size]))# MSE
    
    
    new_row = {
        'modelo':'arima',
        'mae_test':mean_absolute_error(df_test['servicios'], test), 
        'mae_train':mean_absolute_error(df_train['servicios'],df['pronostico'][:size]),  
        'mse_test':mean_squared_error(df_test['servicios'], test), 
        'mse_train':mean_squared_error(df_train['servicios'], df['pronostico'][:size])
    }
    return new_row

    
    
def model_adaline_opt(df):
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

    P = popt
    len_train=int(len(data)*PERCENT_TRAIN)
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
    predlist=[]
    for a in s:
      predlist.append(int(a))
    datalist=[]
    for i in data:
      datalist.append(i)
    errores_train=[]
    for i in datalist[P:len_train+P]:
      for j in predlist[:len_train]:
        error=abs(i-j)
        errores_train.append(error)
    EPE=EPF=round(np.mean(errores_train),2)

    errores_test=[]
    for i in datalist[len_train+P:-1]:
      for j in predlist[len_train:]:
        error=abs(i-j)
        errores_test.append(error)
    EPF=round(np.mean(errores_test),2)
    error_mse_trains=[]
    for i in datalist[P:len_train+P]:
      for j in predlist[:len_train]:
        error=(abs(i-j))**2
        error_mse_trains.append(error)
    mse_train=round(np.mean(error_mse_trains),2)

    error_mse_tests=[]
    for i in datalist[len_train+P:-1]:
      for j in predlist[len_train:]:
        error=(abs(i-j))**2
        error_mse_tests.append(error)
    mse_test=round(np.mean(error_mse_tests),2)

    print("Mean Absolute error test: {}".format(EPF))
    print("Mean Absolute error train: {}".format(EPE))
    print("Mean Squared error test: {}".format(mse_test))
    print("Mean Squared error train: {}".format(mse_train))
    plt.figure(figsize=(14, 5))
    plt.plot(datalist[P:],'.-k')
    plt.title("Ajuste del modelo Adaline", fontdict=None, size=18)
    plt.plot(predlist[:-1], '.-r')
    plt.grid()

    plt.vlines([int(len(data)*PERCENT_TRAIN)], ls='--', color='blue', ymin=min(s), ymax=max(s))
    plt.show()
    
    new_row = {
        'modelo':'adaline',
        'mae_test':EPF, 
        'mae_train':EPE,  
        'mse_test':mse_test, 
        'mse_train':mse_train
    }
    return new_row

###############################################################################################################################333333333    
def model_adaline_optimizado(df):
  optimal_EPF=None
  ini=int(input("Seleccione el número de inicio para probar P: "))
  fin=int(input("Seleccione el número de fin para probar P: "))
  global popt
  for lm in range(ini,fin+1):
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

      P = lm
      len_train=int(len(data)*PERCENT_TRAIN)
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
      predlist=[]
      for a in s:
        predlist.append(int(a))
      datalist=[]
      for i in data:
        datalist.append(i)
      errores_train=[]
      for i in datalist[P:len_train+P]:
        for j in predlist[:len_train]:
          error=abs(i-j)
          errores_train.append(error)
      EPE=round(np.mean(errores_train),2)

      errores_test=[]
      for i in datalist[len_train+P:-1]:
        for j in predlist[len_train:]:
          error=abs(i-j)
          errores_test.append(error)
      EPF=round(np.mean(errores_test),2)
      if optimal_EPF is None or EPF < optimal_EPF:
        optimal_EPF = EPF
        popt=lm
      print("Con P = {} el error abs promedio de entrenamiento: {}, y el error abs promedio de pronóstico: {}".format(lm,EPE,EPF))
  print("El P óptimo es: {}".format(popt))
  return model_adaline_opt(df)


def modelo_perceptron_multicapa(df):
    

    scaler = MinMaxScaler() # crea el transformador
    df_train_scaled = scaler.fit_transform(np.array(df.servicios).reshape(-1, 1)) # escala la serie  # z es un array de listas como efecto
    df_train_scaled = [u[0] for u in df_train_scaled]# z es un array de listas como efecto
    data_d1 = [df_train_scaled[t] - df_train_scaled[t - 1] for t in range(1, len(df_train_scaled))] # Se remueve la tendencia
    data_d1d12 = [data_d1[t] - data_d1[t - 7] for t in range(7, len(data_d1))] # Para remover la componente cíclica cada 8 días
    
    df_train = data_d1d12[0:int(len(data_d1d12)*PERCENT_TRAIN)]
    df_test = data_d1d12[int(len(data_d1d12)*PERCENT_TRAIN):]

    #Se contruye una matriz de regresores, Ya que la implementación dispobible en sklearn es para modelos de regresión, se debe armar una matrix donde las variables independientes son zt−1, …, zt−P y la variable dependiente es zt.

    P = popt#int(input("Ingrese el valor p: "))
    X_train_regr = []
    X_test_regr = []
    X_regr = []

    for t in range(P - 1, len(data_d1d12) - 1):
        X_regr.append([data_d1d12[t - n] for n in range(P)])
        
    for t in range(P - 1, len(df_train) - 1):
        X_train_regr.append([df_train[t - n] for n in range(P)])
        
    for t in range(P - 1, len(df_test) - 1):
        X_test_regr.append([df_test[t - n] for n in range(P)])

    d = df_train[P:]

    #Modelo de percdeptron multicalpa

    H = 2 # Se escoge arbitrariamente el numero de neuronas
    np.random.seed(123456)

    mlp = MLPRegressor(
        hidden_layer_sizes=(H, ),
        #activation = 'logistic',
        activation ='tanh',
        learning_rate = 'adaptive',
        momentum = 0.0,
        learning_rate_init = 20,
        max_iter = 100000)

    mlp.fit(X_train_regr, d)   # Entrenamiento

    y_d1d12_m2 = mlp.predict(X_regr)  # Pronostico
    y_d1d12_m2_test =  mlp.predict(X_test_regr)

    #  transformaciones inversas a las realizadas
    y_d1d12_m2 = data_d1d12[0:P] + y_d1d12_m2.tolist()
    
    y_d1d12_m2_test = df_test[0:P] + y_d1d12_m2_test.tolist()

    y_d1_m2 = [y_d1d12_m2[t] + data_d1[t] for t in range(len(y_d1d12_m2))] # inversa de la remoción del ciclo 
    y_d1_m2 = data_d1[0:7] + y_d1_m2

    y_m2 = [y_d1_m2[t] + df_train_scaled[t] for t in range(len(y_d1_m2))] # agregamos la tendencia 
    y_m2 = [df_train_scaled[0]] + y_m2

    y_m2 = scaler.inverse_transform([[u] for u in y_m2]) # Desescalamos los datos 
    y_m2 = [u[0] for u in y_m2.tolist()] 
    
    print("Mean absolute error: %.2f" % mean_absolute_error(df.servicios[:-len(df_test)], y_m2[:-len(df_test)]))
    print("Mean_absolute_error: %.2f" % mean_absolute_error(df.servicios[-len(df_test):],  y_m2[-len(df_test):]))

    print("Mean squared error: %.2f" % mean_squared_error(df.servicios[:-len(df_test)], y_m2[:-len(df_test)]))# MSE
    print("Mean squared error: %.2f" % mean_squared_error(df.servicios[-len(df_test):], y_m2[-len(df_test):]))# MSE

    #print("Variance score: %.2f" % r2_score(df.servicios, y_m2)) # R2
    #print("Variance score: %.2f" % r2_score(df_test,  mlp.predict(df_test))) # R2

    plt.figure(figsize=(14, 5))
    plt.plot(df.servicios, ".-k")
    plt.plot(y_m2, "-r")
    plt.title("Ajuste del Modelo Perceptrón Multicapa", fontdict=None, size=18)
    plt.plot([len(df.servicios) - len(df_test), len(df.servicios) - len(df_test)], [min(df.servicios), max(df.servicios)], "--", linewidth=2)
    plt.grid()
          
    new_row = {
        'modelo':'perceptron_multicapa',
        'mae_train':mean_absolute_error(df.servicios[:len(df_train)], y_m2[:len(df_train)]), 
        'mae_test':mean_absolute_error(df.servicios[-len(df_test):],  y_m2[-len(df_test):]),  
        'mse_test': mean_squared_error(df.servicios[:-len(df_test)], y_m2[:-len(df_test)]), 
        'mse_train':mean_squared_error(df.servicios[:len(df_train)], y_m2[:len(df_train)])
    }
    return new_row


def modelo_perceptron_multicapa_sin_transformaciones(df):
    P = popt#int(input('Ingrese valor p: '))

    def computar_modelo():
        n = np.random.rand(len(df)) < PERCENT_TRAIN
        df_train = df.servicios[n]
        df_test = df.servicios[~n]
        #size = int(len(df)*PERCENT_TRAIN)
        #df_train, df_test=df[0:size],df[size:len(df)]

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

        n = np.random.rand(len(df)) < PERCENT_TRAIN
        df_train = df.servicios[n]
        df_test = df.servicios[~n]
        #size = int(len(df)*PERCENT_TRAIN)
        #df_train, df_test=df[0:size],df[size:len(df)]

        #P = 8

        e_train = df_train[P:]
        e_y_m1_train = y_m1[:-len(df_test)]
        e_y_m1_test = y_m1[len(e_y_m1_train):]

        print("Mean absolute error: %.2f" % mean_absolute_error(e_train, e_y_m1_train)) 
        print("Mean_absolute_error: %.2f" % mean_absolute_error(df_test, e_y_m1_test))
        print("Mean squared error: %.2f" % mean_squared_error(e_train, e_y_m1_train))# MSE
        print("Mean squared error: %.2f" % mean_squared_error(df_test, e_y_m1_test))# MSE
        #print("Variance score: %.2f" % r2_score(e_train, e_y_m1_train)) # R2
        #print("Variance score: %.2f" % r2_score(df_test, e_y_m1_test)) # R2
        
        return mean_absolute_error(e_train, e_y_m1_train), mean_absolute_error(df_test, e_y_m1_test), mean_squared_error(e_train, e_y_m1_train), mean_squared_error(df_test, e_y_m1_test)

    def graficar_modelo():

        y_m1 = computar_modelo()

        n = np.random.rand(len(df)) < PERCENT_TRAIN
        df_train = df.servicios[n]
        df_test = df.servicios[~n]
        #size = int(len(df)*PERCENT_TRAIN)
        #df_train, df_test=df[0:size],df[size:len(df)]

        #P = 8

        plt.figure(figsize=(14, 5))
        plt.plot(df.servicios, ".-k")
        plt.grid()
        plt.plot([None] * P + y_m1, "-r")
        plt.title("Ajuste del Modelo Perceptrón Multicapa", fontdict=None, size=18)
        plt.plot([len(df.servicios) - len(df_test), len(df.servicios) - len(df_test)], [min(df.servicios), max(df.servicios)], "--", linewidth=2)

    graficar_modelo()
    mae_train, mae_test, mse_train, mse_test = computar_metricas()
    new_row = {
        'modelo':'perceptron_multicapa_sin_transformaciones',
        'mae_test':mae_test, 
        'mae_train':mae_train,  
        'mse_test':mse_test, 
        'mse_train':mse_train
    }
    return new_row
