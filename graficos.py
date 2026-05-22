#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 15:03:38 2026

Graficos 

@author: emer
"""

# %% librerias e importar variables del importar_dataser

import pandas as pd
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from dotenv import load_dotenv

load_dotenv()

work_dir = os.getenv('WORK_DIR')
if work_dir is None:
    raise ValueError('falta el .env o WORK_dir')
    
sys.path.append(work_dir)

from Importar_dataset import dataset_, dataset_ml

# %% graficos y resumen en html

print("html")
profile = ProfileReport(dataset_ml, title="Reporte EDA Nowcasting", tsmode=True)
profile.to_file(os.path.join(work_dir, "reporte_exploratorio.html"))

# %% graficos en seaborn

variables_continuas = ['IPMC', 'mwh_comercial', 'export_fob', 'ipc_us', 'imae']

print("Generando Pairplot de Seaborn...")
sns.pairplot(dataset_ml[variables_continuas].dropna(), diag_kind='kde')
plt.show()

#mapa de correlacion

plt.figure(figsize=(10, 8))
#correlacion de Pearson y anotamos los valores
matriz_corr = dataset_ml[variables_continuas].corr()
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt=".3f", vmin=-1, vmax=1)
plt.title('Matriz de Correlacion')
plt.show()

"""
Series altamente correlacionadas caracteriticas
IPMC, mwh_comercial, diesel, fuel-oil, export_fob , import_cif (es no estacionario --va-
rianza y media no constantes pero si estacional), ipc_us son no estacionarios

mwh_industria es estacionario

el html usa el test Dickey-Fuller h_0: existe una raíz unitaria, no estacionaria
"""