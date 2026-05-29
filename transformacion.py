#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 16:35:04 2026

aqui solo hago transformaciones 

@author: emer
"""

# %% librerias e importar variables del importar_dataset

import pandas as pd
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from ydata_profiling import ProfileReport
from statsmodels.tsa.stattools import adfuller
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

load_dotenv()

work_dir = os.getenv('WORK_DIR')
if work_dir is None:
    raise ValueError('falta el .env o WORK_dir')
    
sys.path.append(work_dir)

from Importar_dataset import dataset_, dataset_ml

# %% tranformar los datos y validar supuestos

#usar otro nombre para no sobreescribir
df_eco = dataset_.copy()
df_ml = dataset_ml.copy()

#convertir el dataframe de pandas a lista de python
columnas_continuas = df_ml.columns.tolist()
#.tolist convierte el array en una lista de python


#matrices para transformacion
matrices = [df_eco, df_ml]

#tranformacion log -- tasa de variacion anual (menos ipc y mwh_comercial)
for variablex in matrices:
    for col in columnas_continuas:
        if col in ['ipc_us', 'mwh_comercial']:
            variablex[col + '_ln'] = (np.log(variablex[col]) - np.log(variablex[col].shift(1))) * 1200
            #lo que se hara es una tasa intermensual
            # tal que Δln(ipc) = (ln(ipcₜ) - ln(ipcₜ-1)) (100 #para hacerlo %) (12 #meses) 
        else:
            #diff algoritmica de las columnas
            variablex[col + '_ln'] = (np.log(variablex[col]) - np.log(variablex[col].shift(12))) * 100
            # [col] es el nombre de la varible dentro de la matriz
            # np.log(variablex[col]) toma el mes y le aplilca ln 
            # np.log(variablex[col].shift(12)) retrocede 12 obs
            # se restan y para obtener un % se multiplica por 100

#filtrar y eliminar los 12 nan        
df_eco_ln = df_eco.filter(regex='_ln|Dummy').dropna()
#ragex indica no buscar la palabra exacta
# | indica "o" 

df_ml_ln = df_ml.filter(like='_ln').dropna()
#filtra y guarda las variables que tiene _ln, elemina los nan

print("\nResumen")
print(f"Meses perdidos por el rezago: {len(df_ml) - len(df_ml_ln)}")
print(f"Dimension matriz de ml: {df_ml_ln.shape}")
print(f"Dimension matriz de econometria: {df_eco_ln.shape}")
print(f"Observaciones: {df_ml_ln.index.size}")

# %% puebra de raiz unitaria

raiz_ml = df_ml_ln.copy()

print('\nTest Dickey fuller')
def prueba_dickey_fuller(dataframe, criterio='BIC'):
    resultados_temporales = []

    for name in dataframe.columns:
        serie_raiz = dataframe[name]
        adf_test = adfuller(serie_raiz, autolag=criterio)
        p_value = adf_test[1]
        resultados_temporales.append({
            'variable': name,
            'p-value': round(p_value, 4),
        })
        
    return pd.DataFrame(resultados_temporales)
            

tabla_adf = prueba_dickey_fuller(dataframe=raiz_ml)

print(tabla_adf)
print('\nLas variables son tasas de variancion anual')
print('\nPara ipc_us y mwh_comercial es tasa de variacion intermesual')

print("\nresumen")
print(f"matriz ML: {df_ml_ln.shape}")
print(f"matriz Econometria: {df_eco_ln.shape}")