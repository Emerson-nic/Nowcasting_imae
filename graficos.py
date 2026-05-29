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
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.signal import periodogram
from ydata_profiling import ProfileReport
from dotenv import load_dotenv

load_dotenv()

work_dir = os.getenv('WORK_DIR')
if work_dir is None:
    raise ValueError('falta el .env o WORK_dir')
    
sys.path.append(work_dir)

from Importar_dataset import dataset_, dataset_ml
from transformacion import df_ml_ln, df_eco_ln

# %% graficos y resumen en html

print("html para dataset sin transfomar")
profile = ProfileReport(dataset_ml, title="Reporte EDA Nowcasting", tsmode=True)
profile.to_file(os.path.join(work_dir, "reporte_exploratorio.html"))

# %% graficos en seaborn sin transformar

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

# %% graficos y resumen transformados

print("html para dataset transfomados")
profile = ProfileReport(df_ml_ln, title="Reporte EDA Nowcasting", tsmode=True)
profile.to_file(os.path.join(work_dir, "reporte_exploratorio_transformados.html"))

# %% ver tendencia

#grafico de tasas vs niveezs normales
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

#serie cruda
axes[0].plot(dataset_['imae'].dropna().index, dataset_['imae'].dropna(), color='black')
axes[0].set_title('IMAE: Serie Original en Niveles (Presencia de Tendencia y Estacionalidad)', fontsize=12)
axes[0].grid(True, linestyle=':', alpha=0.6)

#serie trans
axes[1].plot(df_ml_ln['imae_ln'].index, df_ml_ln['imae_ln'], color='blue')
axes[1].set_title('IMAE: Transformación Estacionaria (Tasa de Variación)', fontsize=12)
axes[1].grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()

#acf
lags = 36 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(df_ml_ln['imae_ln'].dropna(), lags=lags, ax=axes[0], title='ACF- IMAE I(0)')
plot_pacf(df_ml_ln['imae_ln'].dropna(), lags=lags, ax=axes[1], title='PACF - IMAE I(0)', method='ywm')

for ax in axes:
    ax.grid(True, linestyle=':', alpha=0.6)
    #marcar rezagos estacionales (12, 24, 36)
    ax.set_xticks(np.arange(0, lags+1, 12)) 

plt.tight_layout()
plt.show()

#periodo grama
frecuencias, densidad_espectral = periodogram(df_ml_ln['imae_ln'].dropna(), fs=1) # fs=1 por ser datos mensuales

plt.figure(figsize=(10, 5))
plt.plot(frecuencias, densidad_espectral, color='purple', linewidth=1.5)
plt.title('Periodograma (Densidad Espectral) - IMAE I(0)', fontsize=12)
plt.xlabel('Frecuencia ($f$)', fontsize=10)
plt.ylabel('Densidad', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)

# Identificar y anotar los 3 ciclos más fuertes
indices_top3 = np.argsort(densidad_espectral)[-3:][::-1]
for idx in indices_top3:
    f_pico = frecuencias[idx]
    if f_pico > 0:
        periodo = 1 / f_pico
        plt.axvline(x=f_pico, color='red', linestyle='--', alpha=0.5)
        plt.text(f_pico, densidad_espectral[idx], f' T={periodo:.1f} meses', color='black', fontsize=9)

plt.tight_layout()
plt.show()