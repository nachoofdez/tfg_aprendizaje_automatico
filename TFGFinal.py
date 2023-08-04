import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn import tree
from statsmodels.miscmodels.ordinal_model import OrderedModel
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2


'''
Extraccion de los datos y analisis previo
'''
data = pd.read_csv('train.csv') #Lectura de los datos
num_filas = data.shape[0]  # Número de filas
num_columnas = data.shape[1]  # Número de columnas
#Para ver todas las variables
pd.set_option('display.max_columns', None)

with open('analisis_datos.txt','w') as f:  
    #Primeros 8 ejemplos
    f.write(data.head(8).to_string(index=False))
    f.write("\n")
    #Tabla resumen descriptivo variables explicativas
    f.write(data.describe(include="all").to_string(index=False))
    f.write("\n")
    #Descripción estructura del data frame y valores nulos.
    data.info(buf=f)
    f.write("\n")
    #Número de datos únicos por variable explicativa
    nunique_values = data.nunique()
    f.write(nunique_values.to_string())
    f.write("\n")

#Semilla datos aleatorios
np.random.seed(17)

#Crear graficos de linea entre variable explicativa y respuesta
def graficos_variables(variable):
    media_rango_precios = data.groupby(variable)['price_range'].mean()
    plt.plot(media_rango_precios.index, \
             media_rango_precios, marker='o')
    plt.xlabel(variable)
    plt.ylabel('price_range')
    plt.show()

# Separacion de los datos en variable respuesta y explicativa
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

#Normalizadores y Variables a normalizar
columns_to_scale = ['battery_power', 'clock_speed', 'fc','int_memory',\
                    'm_dep','mobile_wt','n_cores','pc','px_height',\
                    'px_width','ram','sc_h','sc_w','talk_time']

scaler = StandardScaler()
scaler2=MinMaxScaler()

# Division datos de entrenamiento y test para validacion cruzada 5x2
CV = RepeatedKFold(n_splits=2, n_repeats=5, random_state=5)

#kfold para la seleccion de hiperparametros en GridSearchCV
kfold = KFold(n_splits=5,shuffle=True, random_state=23)

#Definicion auxiliar para la metrica de evaluacion MARE
def mdp(y_test, y_pred):
        diferencia_abs = np.abs(y_test - y_pred)
        suma_dif =  np.sum(diferencia_abs)
        media_diferencias = suma_dif / len(diferencia_abs)
        return media_diferencias

#Metricas que se van a utilzar en seleccion hiperparametros    
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'macro_f1': make_scorer(f1_score, average='macro'),
    'mdp': make_scorer(mdp, greater_is_better=False)
}

#Funcion auxiliar para hacer la media de los resultados de CV
def suma_arrays(lista_arrays):
    n_arrays = len(lista_arrays)
    suma = np.zeros_like(lista_arrays[0])  
    for array in lista_arrays:
        suma += array
    suma /= n_arrays
    return suma

#Funciones auxiliares preparacion datos para gráficas
def partir_lista1(lista, n):
    sub_listas = [[] for _ in range(n)]
    for i, elemento in enumerate(lista):
        sub_listas[i % n].append(elemento)
    return sub_listas

def partir_lista2(lista, n):
    sub_listas = []
    for i in range(0, len(lista), n):
        sub_listas.append(lista[i:i+n])
    return sub_listas

#Crear las graficas evaluacion-hiperparametros segun el algoritmo
def crear_grafica(x,y,xlabel, ylabel, title):
    plt.plot(x, y, color='red')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
def crear_grafica_k_NN(x, y, xlabel, ylabel, title):
    y_pesos=partir_lista1(y, 2)
    y_uniform=y_pesos[0]
    y_distance=y_pesos[1]
    plt.plot(x, y_uniform, color='blue', label='votacion uniforme')
    plt.plot(x, y_distance, color='red', label='votacion ponderada')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

colores = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta',\
           'yellow', 'black', 'gray', 'brown', 'pink', 'teal', 'navy', \
           'olive', 'salmon', 'gold', 'indigo', 'lavender', 'lime']

def crear_grafica_arboles_parada(X,y,xlabel,ylabel\
                                 ,title,min_impurity_decrease,\
                                     min_samples_split):
    y_mod=partir_lista2(y, len(min_samples_split))
    for i in range(len(y_mod)):
        plt.plot(X, y_mod[i], color=colores[i], \
                 label=f'beta = {min_impurity_decrease[i]}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

#Dado un algoritmo, te devuelve las evaluaciones medias por hiperparametro
#de los GridSearchCV()
def comparacion_hiperparametros(model,filename,X,y):
    scores_acc_kfold=[]
    scores_f1_kfold=[]
    scores_mdp_kfold=[]


    for train_index, test_index in CV.split(X):
        X_train= X.iloc[train_index]
        y_train= y.iloc[train_index]

        model.fit(X_train,y_train)

        scores_acc = model.cv_results_['mean_test_accuracy']
        scores_f1 = model.cv_results_['mean_test_macro_f1']
        scores_mdp= model.cv_results_['mean_test_mdp']

        scores_acc_kfold.append(scores_acc)
        scores_f1_kfold.append(scores_f1)
        scores_mdp_kfold.append(scores_mdp)

    params=model.cv_results_['params']
    scores_acc = suma_arrays(scores_acc_kfold)
    scores_f1 = suma_arrays(scores_f1_kfold)
    scores_mdp = suma_arrays(scores_mdp_kfold)
    
    with open(filename,'w') as f:
        f.write("\nResultados Modelo:\n")
        f.write("\nExactitud:\n")
        f.write(str(scores_acc))
        f.write("\nF1-Score:\n")
        f.write(str(scores_f1))
        f.write("\nMDP:\n")
        f.write(str(scores_mdp))
        f.write("\nParametros:\n")
        f.write(str(params))
        
    return (scores_acc,scores_f1,scores_mdp)

#Dado un modelo, te devuelve las evaluaciones medias en los tests de 5x2CV
def evaluaciones_CV(model,X,y):
    scores_acc_CV=[]
    scores_f1_CV=[]
    scores_mdp_CV=[]
    
    for train_index, test_index in CV.split(X):
        X_train= X.iloc[train_index]
        y_train= y.iloc[train_index]
        X_test= X.iloc[test_index]
        y_test= y.iloc[test_index]
    
    
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        acc= accuracy_score(y_test, y_pred)
        scores_acc_CV.append(acc)
        f1= f1_score(y_test, y_pred, average='macro')
        scores_f1_CV.append(f1)
        mdp_score= mdp(y_test, y_pred)
        scores_mdp_CV.append(mdp_score)
    
    acc_CV = np.mean(scores_acc_CV)
    f1_CV = np.mean(scores_f1_CV)
    mdp_CV = np.mean(scores_mdp_CV)
    return (acc_CV,f1_CV,mdp_CV)

#Normalizando previamente los conjuntos de entrenamiento
def evaluaciones_CV_norm(model,X,y,normalizador):
    scores_acc_CV=[]
    scores_f1_CV=[]
    scores_mdp_CV=[]
    
    for train_index, test_index in CV.split(X):
        X_train= X.iloc[train_index]
        y_train= y.iloc[train_index]
        X_test= X.iloc[test_index]
        y_test= y.iloc[test_index]
    
        scaler.fit(X_train[columns_to_scale])

        X_train_scaled = X_train.copy()
        X_train_scaled[columns_to_scale] = scaler.\
            transform(X_train_scaled[columns_to_scale])

        X_test_scaled = X_test.copy()
        X_test_scaled[columns_to_scale] = scaler.\
            transform(X_test_scaled[columns_to_scale])
            
        model.fit(X_train_scaled,y_train)
        y_pred=model.predict(X_test_scaled)
        acc= accuracy_score(y_test, y_pred)
        scores_acc_CV.append(acc)
        f1= f1_score(y_test, y_pred, average='macro')
        scores_f1_CV.append(f1)
        mdp_score= mdp(y_test, y_pred)
        scores_mdp_CV.append(mdp_score)
    
    acc_CV = np.mean(scores_acc_CV)
    f1_CV = np.mean(scores_f1_CV)
    mdp_CV = np.mean(scores_mdp_CV)
    return (acc_CV,f1_CV,mdp_CV)

#Para los modelos de regresion:
def evaluaciones_CV_reg(data_reg_final,max_iter):
    scores_acc_CV=[]
    scores_f1_CV=[]
    scores_mdp_CV=[]
    X_sub=data_reg_final.iloc[:,:-1]
    
    for train_index, test_index in CV.split(X_sub):
        X_train, X_test = X_sub.iloc[train_index], X_sub.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model5=OrderedModel(y_train, X_train, distr='logit')
        results=model5.fit(method='nm',maxiter=max_iter)
        predicted = results.model.predict(results.params, X_test)
        y_pred = np.argmax(predicted, axis=1)
        f1=f1_score(y_test, y_pred,average='macro')
        scores_f1_CV.append(f1)
        acc=accuracy_score(y_test, y_pred)
        scores_acc_CV.append(acc)
        mdp_score=mdp(y_test,y_pred)
        scores_mdp_CV.append(mdp_score)
    
    acc_CV = np.mean(scores_acc_CV)
    f1_CV = np.mean(scores_f1_CV)
    mdp_CV = np.mean(scores_mdp_CV)
    return (acc_CV,f1_CV,mdp_CV)
        
#Funcion resultados de regresion logistica en un entrenamiento
def entrenamiento_regresion(columns_to_drop_wald,num_entrenamiento,\
                            maxiter):
    subset_data=data_reg.drop(columns=columns_to_drop_wald)
    X_sub=subset_data.iloc[:,:-1]
    
    train_index, test_index = list(CV.split(X))[num_entrenamiento]
    X_train, X_test = X_sub.iloc[train_index], X_sub.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model5=OrderedModel(y_train, X_train, distr='logit')
    results=model5.fit(method='nm',maxiter=maxiter)
    print(results.summary())
    predicted = results.model.predict(results.params, X_test)
    y_pred = np.argmax(predicted, axis=1)
    f1=f1_score(y_test, y_pred,average='macro')
    acc=accuracy_score(y_test, y_pred)
    mdp_score=mdp(y_test,y_pred)
    scores=(acc,f1,mdp_score)
    
    return scores

#Y normalizando las variables
def entrenamiento_regresion_norm(columns_to_drop_wald,num_entrenamiento,\
                            maxiter,columns_to_scale):
    subset_data=data_reg.drop(columns=columns_to_drop_wald)
    X_sub=subset_data.iloc[:,:-1]
    
    train_index, test_index = list(CV.split(X))[num_entrenamiento]
    X_train, X_test = X_sub.iloc[train_index], X_sub.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    scaler.fit(X_train[columns_to_scale])

    X_train_scaled = X_train.copy()
    X_train_scaled[columns_to_scale] = scaler.\
        transform(X_train_scaled[columns_to_scale])
    X_test_scaled = X_test.copy()
    X_test_scaled[columns_to_scale] = scaler.\
        transform(X_test_scaled[columns_to_scale])
    
    model5=OrderedModel(y_train, X_train_scaled, distr='logit')
    results=model5.fit(method='nm',maxiter=maxiter)
    print(results.summary())
    predicted = results.model.predict(results.params, X_test_scaled)
    y_pred = np.argmax(predicted, axis=1)
    f1=f1_score(y_test, y_pred,average='macro')
    acc=accuracy_score(y_test, y_pred)
    mdp_score=mdp(y_test,y_pred)
    scores=(acc,f1,mdp_score)
    return scores

#Calculo estadistico Bondad de Ajuste Regresion Logistica
def calculo_estadistico_CG(probabilidades,y):
    ordinal_scores = np.sum(probabilidades * np.arange(1, 4+1), axis=1)
    sorted_indices = np.argsort(ordinal_scores)
    probabilidades_sorted=probabilidades[sorted_indices]
    y_sorted = y[sorted_indices]
    probabilidades_groups = np.array_split(probabilidades_sorted, 10)
    y_groups = np.array_split(y_sorted, 10)
    observed_frequencies = []
    expected_frequencies = []
    for g in range(10):
        group_pred = probabilidades_groups[g]
        group_obs = y_groups[g]
        group_pred_t = np.transpose(group_pred)
        group_freq_pred = np.sum(group_pred_t, axis=1)
        group_freq_obs = np.bincount(group_obs, minlength=4)
        expected_frequencies.append(group_freq_pred)
        observed_frequencies.append(group_freq_obs)
    CG=0
    for g in range(10):
        for j in range(4):
            if expected_frequencies[g][j]!=0:
                aux=(observed_frequencies[g][j]-\
                     expected_frequencies[g][j])**2
                aux2=aux/expected_frequencies[g][j]
                CG=CG+aux2
    return CG

'''
-------------COMPARACION HIPERPARAMETROS------------

Modelo 1: Arboles para clasificacion con parada temprana
'''
np.random.seed(27)

param_grid1 = {
    'min_samples_split': [3,5,8,9,10,11,12,15],
    'min_impurity_decrease': [0,0.0001,0.001,0.01],
}

min_samples_split=param_grid1['min_samples_split']
min_impurity_decrease=param_grid1['min_impurity_decrease']

model1=GridSearchCV(estimator=DecisionTreeClassifier(criterion='entropy'\
                                                     ,splitter='best'),\
                    param_grid=param_grid1,\
                    scoring=scoring,\
                    cv=kfold,\
                    refit=False)

(scores_acc,scores_f1,scores_mdp)=\
    comparacion_hiperparametros(model1,'resultados_arbparada.txt',X,y)

#Graficas
fig_acc=plt.figure()
crear_grafica_arboles_parada(min_samples_split, \
                             scores_acc, 'm',\
                                 'Exactitud',\
                                     'Exactitud hiperparametros',\
                                         min_impurity_decrease,\
                                             min_samples_split)
plt.show(fig_acc)

fig_f1=plt.figure()
crear_grafica_arboles_parada(min_samples_split, scores_f1,\
                      'm','F1-Score','F1-score hiperparametros',\
                          min_impurity_decrease,min_samples_split)
plt.show(fig_f1)

fig_mdp=plt.figure()
crear_grafica_arboles_parada(min_samples_split, scores_mdp,'m','MDP',\
                      'MDP hiperparametros',min_impurity_decrease,\
                          min_samples_split)
plt.show(fig_mdp)

'''
Modelo 2: Arboles para clasificacion con poda
'''
np.random.seed(21)

param_grid2 = {  
    'ccp_alpha':[0.0001,0.0005,0.001,0.002,0.0035,\
                 0.005,0.007,0.008,0.009,0.01],
}

ccp_alpha=param_grid2['ccp_alpha']

model2=GridSearchCV(estimator=DecisionTreeClassifier(criterion='entropy'\
                                                     ,splitter='best'),\
                    param_grid=param_grid2,\
                    scoring=scoring,\
                    cv=kfold,\
                    refit=False)

(scores_acc,scores_f1,scores_mdp)=\
    comparacion_hiperparametros(model2,'resultados_arbpoda.txt',X,y)
    
#Graficas
fig_acc=plt.figure()
crear_grafica(ccp_alpha, scores_acc, 'alpha', 'Exactitud', \
              'Exactitud-Valores de alpha')
plt.show(fig_acc)

fig_f1=plt.figure()
crear_grafica(ccp_alpha, scores_f1, 'alpha', 'F1-Score', \
              'F1-Score-Valores de alpha')
plt.show(fig_f1) 

fig_mdp=plt.figure()
crear_grafica(ccp_alpha, scores_mdp, 'alpha', 'MDP', \
              'MDP-Valores de alpha')
plt.show(fig_mdp)

'''
Modelo 3 V1: k-NN (Sin Normalizar Variables)
'''
param_grid3 = {
    'n_neighbors': [1,3,5,7,10,12,15,17,20,22,25,27,30,35],
    'weights': ['uniform','distance'],
}

vecinos=param_grid3['n_neighbors']
pesos=param_grid3['weights']

model3=GridSearchCV(estimator=KNeighborsClassifier(),\
                    param_grid=param_grid3,\
                    scoring=scoring,\
                    cv=kfold,\
                    refit=False)

(scores_acc,scores_f1,scores_mdp)=\
    comparacion_hiperparametros(model3,'resultados_kNN_noNorm.txt',X,y)

#Graficas:
fig_acc=plt.figure()
crear_grafica_k_NN(vecinos, scores_acc, 'Vecinos',\
                   'Exactitud', 'Exactitud hiperparametros')
plt.show(fig_acc)

fig_f1=plt.figure()
crear_grafica_k_NN(vecinos,scores_f1, 'Vecinos',\
                   'F1-Score','F1-Score hiperparametros')
plt.show(fig_f1)

fig_mdp=plt.figure()
crear_grafica_k_NN(vecinos, scores_mdp ,'Vecinos','MDP',\
              'MDP hiperparametros')
plt.show(fig_mdp)

'''
Modelo 3 V2: k-NN (Seleccion Variables Gran Escala No Normalizadas)

(Tras varias pruebas, dejo seleccionadas las variables con las que\
se consiguen mejores resultados. Esta version es la que mejor 
resultados consigue de las 4)
'''
#Selección variables mayor escala
subset_datakNN = data[['battery_power','int_memory','pc','mobile_wt',\
                    'px_height','px_width','ram','price_range']]

X_sub = subset_datakNN.iloc[:, :-1]

param_grid3 = {
    'n_neighbors': [1,3,5,7,10,12,15,17,20,22,25,27,30,35],
    'weights': ['uniform','distance'],
}

vecinos=param_grid3['n_neighbors']
pesos=param_grid3['weights']

model3=GridSearchCV(estimator=KNeighborsClassifier(),\
                    param_grid=param_grid3,\
                    scoring=scoring,\
                    cv=kfold,\
                    refit=False)

(scores_acc,scores_f1,scores_mdp)=\
    comparacion_hiperparametros(model3,'resultados_kNN_NoNormImp.txt',\
                                X_sub,y)

#Graficas:
fig_acc=plt.figure()
crear_grafica_k_NN(vecinos, scores_acc, 'Vecinos',\
                   'Exactitud','Exactitud hiperparámetros')
plt.show(fig_acc)

fig_f1=plt.figure()
crear_grafica_k_NN(vecinos, scores_f1, 'Vecinos',\
                   'F1-Score','F1-Score hiperparámetros')
plt.show(fig_f1)

fig_mdp=plt.figure()
crear_grafica_k_NN(vecinos, scores_mdp,'Vecinos','MDP',\
              'MDP hiperparámetros')
plt.show(fig_mdp)

'''
Modelo 3 V3: k-NN (Variables Normalizadas) 

(He elegido la normalizacion por estandarizacion)
'''
param_grid3 = {
    'kneighborsclassifier__n_neighbors': [20, 30, 40, 50, 75, 100,\
                                          120, 150,175,200],                                   
    'kneighborsclassifier__weights': ['uniform', 'distance'],
}

vecinos=param_grid3['kneighborsclassifier__n_neighbors']
pesos=param_grid3['kneighborsclassifier__weights']

preprocessor = ColumnTransformer(
    transformers=[('scaler', scaler, columns_to_scale)],
    remainder='passthrough')

pipeline = make_pipeline(preprocessor, KNeighborsClassifier())

model3=GridSearchCV(pipeline,\
                    param_grid=param_grid3,\
                    scoring=scoring,\
                    cv=kfold,\
                    refit=False)

(scores_acc,scores_f1,scores_mdp)=\
    comparacion_hiperparametros(model3,'resultados_kNN_Norm.txt',\
                                X,y)

#Graficas
fig_acc=plt.figure()
crear_grafica_k_NN(vecinos, scores_acc, 'Vecinos',\
                   'Exactitud','Exactitud hiperparametros')
plt.show(fig_acc)

fig_f1=plt.figure()
crear_grafica_k_NN(vecinos, scores_f1, 'Vecinos',\
                   'F1-Score', 'F1-Score hiperparametros')
plt.show(fig_f1)

fig_mdp=plt.figure()
crear_grafica_k_NN(vecinos, scores_mdp,'Vecinos','MDP',\
              'MDP hiperparametros')
plt.show(fig_mdp)


'''
Modelo 3 V4: k-NN (Normalizacion Variables con mayor escala)

(Tras varias pruebas, dejo seleccionadas las variables con las que \
se consiguen mejores resultados)
'''
param_grid3 = {
    'kneighborsclassifier__n_neighbors':[1,3,5,7,10,12,15,\
                                         17,20,22,25,27,30,35],                                   
    'kneighborsclassifier__weights': ['uniform', 'distance'],
}

#Escogemos en este caso las variables con las que se consiguen
#mejores resultados 

subset_datakNN2 = data[['battery_power', 'mobile_wt','px_height',\
                        'px_width','ram','price_range']]

X_sub2 = subset_datakNN2.iloc[:, :-1]

columns_to_scale2 = ['battery_power',\
                    'mobile_wt','px_height',\
                    'px_width','ram']
    
vecinos=param_grid3['kneighborsclassifier__n_neighbors']
pesos=param_grid3['kneighborsclassifier__weights']

preprocessor = ColumnTransformer(
    transformers=[
        ('scaler', scaler, columns_to_scale2)
    ],
    remainder='passthrough'  
)

pipeline = make_pipeline(preprocessor, KNeighborsClassifier())

model3=GridSearchCV(pipeline,\
                    param_grid=param_grid3,\
                    scoring=scoring,\
                    cv=kfold,\
                    refit=False)

(scores_acc,scores_f1,scores_mdp)=\
    comparacion_hiperparametros(model3,'resultados_kNN_NormImp.txt',\
                                X_sub2,y)

#Graficas:
fig_acc=plt.figure()
crear_grafica_k_NN(vecinos, scores_acc, 'Vecinos',\
                   'Exactitud', 'Exactitud hiperparametros')
plt.show(fig_acc)

fig_f1=plt.figure()
crear_grafica_k_NN(vecinos, scores_f1, 'Vecinos',\
                   'F1-Score', 'F1-Score hiperparametros')
plt.show(fig_f1)

fig_mdp=plt.figure()
crear_grafica_k_NN(vecinos,scores_mdp,'Vecinos','MDP',\
              'MDP hiperparametros')
plt.show(fig_mdp)

'''
Modelo 4: Redes Neuronales

(Aunque hemos realizado la prueba para todas las posibles combinaciones
 de hiperparametros, dejamos seleccionadas ya las mejores y variamos
solo el número de neuronas, ya que si no es muy costoso 
computacionalmente. El modelo ya hace la creación de los target values
de manera correcta)
'''

param_grid4 = {
    'mlpclassifier__hidden_layer_sizes': [(25),(50),(100),(150)\
                                          ,(200),(250),(300),(350),\
                                              (400),(450)],
    #'mlpclassifier__alpha':[0.0,0.0001,0.001,0.01,0.1],
    #'mlpclassifier__max_iter':[200,300,500,750,1000],
    #'mlpclassifier__learning_rate':['constant','invscaling'],
    #'mlpclassifier__learning_rate_init':[0.0001,0.001],
    #'early_stopping':[True,False],
}

neuronas=param_grid4['mlpclassifier__hidden_layer_sizes']

preprocessor = ColumnTransformer(
    transformers=[
        ('scaler', scaler, columns_to_scale)
    ],
    remainder='passthrough' 
)

pipeline = make_pipeline(preprocessor,\
                         MLPClassifier(max_iter=1000,\
                                       random_state=17,\
                                           learning_rate='constant',\
                                               solver='sgd',\
                                                   activation='logistic',\
                                                       batch_size='auto'))

model4=GridSearchCV(pipeline,\
                    param_grid=param_grid4,\
                    scoring=scoring,\
                    cv=kfold,\
                    refit=False)

(scores_acc,scores_f1,scores_mdp)=\
    comparacion_hiperparametros(model4,'resultados_redes',X,y)
    
#Graficas:
fig_acc=plt.figure()
crear_grafica(neuronas, scores_acc, 'Neuronas','Exactitud', \
              'Exactitud-Numero Neuronas')
plt.show(fig_acc)

fig_f1=plt.figure()
crear_grafica(neuronas,scores_f1, 'Neuronas','F1-Score', \
              'F1-Score-Numero Neuronas')
plt.show(fig_f1)

fig_mdp=plt.figure()
crear_grafica(neuronas,scores_mdp,'Neuronas','MDP', \
              'MDP-Numero Neuronas')
plt.show(fig_mdp)

'''
Regresion logistica Ordinal:
(Tras aplicar backward, dejamos ya seleccionadas las variables
 que vamos a eliminar del modelo en cada entrenamiento)
'''
matriz_correlacion=data.corr()
#Variables con alta correlacion con otras
columns_to_drop_corr = ['three_g','sc_h','fc']
data_reg=data.drop(columns=columns_to_drop_corr)

'''
Entrenamiento 1:
battery_power,mobile_wt,px_height,px_width,ram->(0.983,0.982,0.017)
'''
columns_to_drop_wald1=['touch_screen','dual_sim','wifi','m_dep','n_cores',\
                       'blue','four_g','clock_speed','sc_w','int_memory'\
                           ,'pc','talk_time']
scores=entrenamiento_regresion(columns_to_drop_wald1, 0, 6000)
print(scores)

'''
Entrenamiento 2:
battery_power,int_memory,mobile_wt,pc,px_height,px_width,ram,sc_w\
    ->(0.974,0.974,0.026)
'''
columns_to_drop_wald2=['blue','m_dep','dual_sim','touch_screen','wifi',\
                       'four_g','talk_time','n_cores','clock_speed']
scores=entrenamiento_regresion(columns_to_drop_wald2,1, 6000)
print(scores)

'''
Entrenamiento 3:
battery_power,int_memory,mobile_wt,px_height,px_width,ram\
    ->(0.978,0.978,0.022)
'''

columns_to_drop_wald3=['wifi','blue','four_g','m_dep','clock_speed'\
                       ,'touch_screen','sc_w','dual_sim','talk_time',\
                           'n_cores','pc']
scores=entrenamiento_regresion(columns_to_drop_wald3,2, 6000)
print(scores)

'''
Entrenamiento 4:
battery_power,mobile_wt,px_height,px_width,ram->(0.974,0.973,0.026)
'''
columns_to_drop_wald4=['clock_speed','dual_sim','four_g','m_dep',\
                       'n_cores','int_memory','wifi','pc','sc_w',\
                           'blue','talk_time','touch_screen']
scores=entrenamiento_regresion(columns_to_drop_wald4,3, 6000)
print(scores)

'''
Entrenamiento 5:
battery_power,mobile_wt,px_height,px_width,ram,wifi->(0.973,0.973,0.027)
'''
columns_to_drop_wald5=['clock_speed','blue','int_memory','touch_screen'\
                       ,'dual_sim','m_dep','pc','four_g','n_cores',\
                           'talk_time','sc_w']
scores=entrenamiento_regresion(columns_to_drop_wald5,4, 6000)
print(scores)

'''
Entrenamiento 6:
battery_power,mobile_wt,px_height,px_width,ram->(0.98,0.98,0.02)
'''
columns_to_drop_wald6=['pc','dual_sim','four_g','touch_screen','wifi'\
                       ,'n_cores','m_dep','clock_speed','blue',\
                           'int_memory','talk_time','sc_w']
scores=entrenamiento_regresion(columns_to_drop_wald6,5, 6000)
print(scores)

'''
Entrenamiento 7:
battery_power,mobile_wt,px_height,px_width,ram->(0.973,0.973,0.027)
'''
columns_to_drop_wald7=['touch_screen','m_dep','pc','talk_time','dual_sim'\
                       ,'blue','wifi','sc_w','four_g','n_cores',\
                           'int_memory','clock_speed']
scores=entrenamiento_regresion(columns_to_drop_wald7,6, 6000)
print(scores)

'''
Entrenamiento 8:
battery_power,int_memory,mobile_wt,n_cores,px_height,px_width,ram,sc_w\
    ->(0.94,0.94,0.06)
'''
columns_to_drop_wald8=['m_dep','blue','four_g','wifi','dual_sim','pc',\
                       'touch_screen','talk_time','clock_speed']
scores=entrenamiento_regresion(columns_to_drop_wald8,7, 8000)
print(scores)

'''
Entrenamiento 9:
battery_power,int_memory,mobile_wt,pc,px_height,px_width,ram\
    ->(0.977,0.976,0.023)
'''
columns_to_drop_wald9=['touch_screen','talk_time','blue','dual_sim',\
                       'four_g','clock_speed','m_dep','wifi','n_cores',\
                           'sc_w']
scores=entrenamiento_regresion(columns_to_drop_wald9,8, 8000)
print(scores)

'''
Entrenamiento 10:
battery_power,clock_speed,mobile_wt,px_height,px_width,ram,sc_w,talk_time\
    ->(0.94,0.94,0.06)
'''
columns_to_drop_wald10=['wifi','n_cores','touch_screen','four_g',\
                        'dual_sim','int_memory','blue','m_dep',\
                            'pc']
scores=entrenamiento_regresion(columns_to_drop_wald10,9, 8000)
print(scores)


'''
Prueba con variables normalizadas:
    
(Llegamos a exactitud de 0.82, muy alejado de los 
resultados sin normalizar, asique no hacemos mas
pruebas)
'''

columns_to_drop_wald_norm1=['n_cores','px_width','sc_w','blue','mobile_wt',\
                            'px_height','wifi','clock_speed','int_memory',\
                                'pc','four_g','touch_screen','talk_time',\
                                    'dual_sim','m_dep']
columns_to_scale1=['battery_power',
                    'ram']
    
scores=entrenamiento_regresion_norm(columns_to_drop_wald_norm1,0, 6000\
                                    ,columns_to_scale1)
print(scores)


'''
-------COMPARACION MEJORES MODELOS--------

Resultados CV Modelo 1:
'''
modelo_mejor1=DecisionTreeClassifier(criterion='entropy',splitter='best',\
                                     min_samples_split=9,
                                     min_impurity_decrease=0.001) 
scores_modelo_mejor1=evaluaciones_CV(modelo_mejor1,X,y)
with open('evaluaciones_finales.txt','w') as f:
    f.write("\n Resultados mejor árbol con parada:\n")
    f.write(str(scores_modelo_mejor1))

#Grafico del árbol con todos los datos
modelo_mejor1.fit(X,y)
class_names = list(map(str, modelo_mejor1.classes_))
plt.figure(figsize=(20, 16))
tree.plot_tree(modelo_mejor1, filled=True, fontsize=10,\
               feature_names=X.columns, class_names=class_names)
plt.show()

# Indicadores de importancia de las variables explicativas (ordenados)
importance = modelo_mejor1.feature_importances_
feature_importances = list(zip(importance, X.columns))
feature_importances_sorted = sorted(feature_importances, reverse=True)
for importance, feature in feature_importances_sorted:
    if importance>0:
        print(f"{feature}: {importance}")

'''
Resultados CV Modelo 2:
'''
modelo_mejor2=DecisionTreeClassifier(criterion='entropy',splitter='best',\
                                     ccp_alpha=0.007)  
scores_modelo_mejor2=evaluaciones_CV(modelo_mejor2,X,y)
with open('evaluaciones_finales.txt','w') as f:
    f.write("\n Resultados mejor arbol con poda:\n")
    f.write(str(scores_modelo_mejor2))

#Gráfico del árbol con todos los datos
modelo_mejor2.fit(X,y)
class_names = list(map(str, modelo_mejor2.classes_))
plt.figure(figsize=(20, 16))
tree.plot_tree(modelo_mejor2, filled=True, fontsize=10,\
               feature_names=X.columns, class_names=class_names)
plt.show()

# Indicadores de importancia de las variables explicativas (ordenados)
importance = modelo_mejor2.feature_importances_
feature_importances = list(zip(importance, X.columns))
feature_importances_sorted = sorted(feature_importances, reverse=True)
for importance, feature in feature_importances_sorted:
    if importance>0:
        print(f"{feature}: {importance}")

'''
Resultados CV Modelo 3
'''
modelo_mejor3=KNeighborsClassifier(weights='distance',n_neighbors=20)
subset_datakNN = data[['battery_power','int_memory','pc','mobile_wt',\
                    'px_height','px_width','ram','price_range']]
X_sub = subset_datakNN.iloc[:, :-1]
scores_modelo_mejor3=evaluaciones_CV(modelo_mejor3,X_sub,y)
with open('evaluaciones_finales.txt','w') as f:
    f.write("\n Resultados mejor modelo k-NN:\n")
    f.write(str(scores_modelo_mejor3))

'''
Resultados CV Modelo 4
'''
modelo_mejor4=MLPClassifier(max_iter=1000,random_state=17,\
                            learning_rate='constant', solver='sgd',\
                                activation='logistic',batch_size='auto',\
                                    hidden_layer_sizes=(400))   
scores_modelo_mejor4=evaluaciones_CV_norm(modelo_mejor4,X,y,scaler)
with open('evaluaciones_finales.txt','w') as f:
    f.write("\n Resultados mejor red neuronal MLP:\n")
    f.write(str(scores_modelo_mejor4))

'''
Resultados CV Modelo 5
'''
columns_to_drop_final=['three_g','sc_h','fc','touch_screen','dual_sim',\
                       'wifi','m_dep','n_cores','blue','four_g',\
                           'clock_speed','sc_w','int_memory'\
                           ,'pc','talk_time']
data_reg_final=data.drop(columns=columns_to_drop_final)

scores_modelo_mejor5=evaluaciones_CV_reg(data_reg_final,8000)

#Entrenamos el mejor modelo con todos los datos
X_reg_final=data_reg_final.iloc[:, :-1]
modelo_mejor5=OrderedModel(y, X_reg_final, distr='logit')
results_final=modelo_mejor5.fit(method='nm',maxiter=6000)
predicted_final = results_final.model.predict(results_final.params,\
                                              X_reg_final)
y_pred = np.argmax(predicted_final, axis=1)
#Test Bondad de Ajuste
CG=calculo_estadistico_CG(predicted_final,y)
df = (10 - 2) * (4 - 1) + (4 - 2)
critical_value = chi2.ppf(1 - 0.05, df)

#Calculo Odds ratio
coeficientes = results_final.params
odds_ratios = np.exp(coeficientes)


with open('evaluaciones_finales.txt','w') as f:
    f.write("\n Evaluaciones mejor modelo Regresion Logistica\
            Ordinal:\n")
    f.write(str(scores_modelo_mejor5))
    f.write("\n Información relevante modelo RLO:\n")
    f.write(str(results_final.summary()))
    f.write("\n Coeficientes:\n")
    f.write(str(coeficientes))
    f.write("\nBondad de Ajuste:\n")
    if CG < critical_value:
        f.write("Hay una buena bondad de ajuste.\
                No se rechaza la hipótesis nula.")
    else:
        f.write("No hay una buena bondad de ajuste. \
                Se rechaza la hipótesis nula.")
    f.write("\n Odds Ratio:\n")
    f.write(str(odds_ratios))


