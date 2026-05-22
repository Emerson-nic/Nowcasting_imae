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

#tranformacion log -- tasa de variacion anual 
for variablex in matrices:
    for col in columnas_continuas:
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

#por correlacion se elimina mwh_comercial
#el ipc debe diferenciarse de nuevo

# %% 2da diff ipc

for matriz in [df_eco_ln, df_ml_ln]:
    matriz['ipc_us_ln'] = matriz['ipc_us_ln'] - matriz['ipc_us_ln'].shift(1)

#eliminar mwg_comercial_ln
df_eco_ln = df_eco_ln.drop(columns=['mwh_comercial_ln'])
df_ml_ln = df_ml_ln.drop(columns=['mwh_comercial_ln'])

# %% despues de eliminar mwh_comercial_ln
#eliminar los na
df_eco_ln = df_eco_ln.dropna()
df_ml_ln = df_ml_ln.dropna()

#cambio de nombre
for matriz in [df_eco_ln, df_ml_ln]:
    matriz.rename(columns={'ipc_us_ln': 'ipc_us_ln_i2'}, inplace=True)

print("\nresumen")
print(f"matriz ML: {df_ml_ln.shape}")
print(f"matriz Econometria: {df_eco_ln.shape}")

#prueba de raiz unitaria
print('\nestacionariedad ipc I(2)\n')
tabla_adf1 = prueba_dickey_fuller(dataframe=df_ml_ln, criterio='BIC')
print(tabla_adf1)
