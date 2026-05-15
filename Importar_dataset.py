#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 13:59:44 2026

Este archivo se usara para importar datos como una funcion y limpieza

librerias
padas
python-dotenv
openpyxl
numpy

@author: emer
"""
# %% liberias

import pandas as pd
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()

work_dir = os.getenv('WORK_DIR')
if work_dir is None:
    raise ValueError('falta el .env o WORK_dir')
#%% importar dataset

#importar dataset
mamitas_pueblas = 'dataset.xlsx'
ruta = os.path.join(work_dir, mamitas_pueblas)

#indice
#'index_col=0' para fechas en la primera columna
# 'parse_dates=True'formato fecha
dataset = pd.read_excel(ruta, sheet_name='Hoja1', index_col=0, parse_dates=True)
dataset.index = pd.to_datetime({
    'year': dataset.index.year,
    'month': dataset.index.day,   
    'day': dataset.index.month    
})

#verifica los datos
print('\n verificacion de datos\n')
print(dataset.info())

print('\n Las primeras 10 observaciones\n')
print(dataset.head(10))
#%% observacion de datos y limpieza 

#frecuencia
frecuencia = pd.infer_freq(dataset.index)

print(f"\nFrecuencia inferida por Pandas: {frecuencia}")

#tipos de datos
#.select_dtypes filtra por columnas
#include=['object'] obtiene los strings
#.columns extrae los nombres de las columnas
#.tolist() extra el indice de forma [] lo hace mas facil de interpretar
tipos_object = dataset.select_dtypes(include=['object']).columns#.tolist()

print(f'\nlista de string del dataset {tipos_object}')

#contar los nan
totalnan = dataset.isna().sum()

#len(dataset) es el tamaño de la muestra, el len() devuelve la longitud de un objeto
porcentajenan = (totalnan / len(dataset)) * 100

print(f"\n \nTotal de NaN \n{totalnan} \n\n Porcentaje de NaN \n{porcentajenan}")

#contar los 0
cero = (dataset == 0 ).sum()

print(f" \nTotal de ceros \n{cero} \n")
#%% imputar y dataset finales

#eliminar la ultima fila porque existen nan por 1 obs en la remesas
dataset_ = dataset.iloc[:-1].copy()

#imputar o rellenar el nan de ipc_us con el metodo de 'time'
dataset_['ipc_us'] = dataset_['ipc_us'].interpolate(method='time')

#eliminar si quedan espacios en blancos
dataset_.columns = dataset_.columns.str.strip()

#dataset para ml
print(dataset_.columns.tolist())
dataset_ml = dataset_.drop(columns=['Dummy_2008', 'Dummy_2018', 'Dummy_2020'])

#dataset_ es para econometria ordinaria