import os

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2  # aggiungi in tesi
#from sklearn.feature_selection import mutual_info_classif
#from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.preprocessing import StandardScaler
import sys  # utilizzato per il reindirizzamento dell'output a file esterni
import category_encoders as ce  # Il Target Encoding è una tecnica di codifica utilizzata per convertire variabili
# categoriali in variabili numeriche, sfruttando la relazione tra ogni categoria e la variabile target.
#from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from collections import Counter


import sklearn
import category_encoders
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler





def detect_and_convert_to_boolean(df):
    # Rileva automaticamente le colonne che possono essere convertite in booleano
    # e le converte gestendo diversi formati di valori booleani.

    for col in df.columns:
        unique_values = df[col].dropna().unique() # rimuove eventuali valori mancanti e prende tutti gli altri vslori
        # un'unica volta
        # Verifica se i valori unici nella colonna sono compatibili con valori booleani
        possible_boolean_values = {'si', 'no', True, False, 1, 0, 'True', 'False', '1', '0'}
        # Se tutti i valori unici sono compatibili con valori booleani, converte la colonna
        if set(unique_values).issubset(possible_boolean_values): # issubset verifica se tutti gli elementi di un insieme
            # sono contenuti in un altro insieme
            # la colonna col viene mappata tramite la funzione map i valori a True/False
            df[col] = df[col].map({
                'si': True, 'no': False,
                'True': True, 'False': False,
                '1': True, '0': False,
                True: True, False: False,
                1: True, 0: False
            })

            # Converte la colonna in tipo booleano
            df[col] = df[col].astype('bool')

        return df


def fill_missing_values(df):
    # Itera su ogni colonna del DataFrame
    for col in df.columns:
        if df[col].dtype == 'object' or 'bool':  # Se il tipo di dati della colonna è 'object' oppure 'bool'
            # Sostituisci i valori mancanti con la moda (valore più frequente)
            mode_value = df[col].mode()[0]  # mode() restituisce una serie, prende i valori che compaiono di piu nella
            # colonna, se compaiono due valori con una stessa frequenza, mode li restituisce entrambi
            df[col] = df[col].replace(np.nan, mode_value) # sostituisco gli nan con la moda
        else:  # Se la colonna è numerica
            # Sostituisco i valori mancanti con la media
            mean_value = df[col].mean()
            df[col] = df[col].replace(np.nan, mean_value)
    return df


def plot_barplot(df_top, score_column, title, xlabel, ylabel, percentile, filename=None, save_plot=False):
    sns.barplot(x=score_column, y='Feature', data=df_top)
    # Aggiungi i valori dei punteggi sopra le barre
    for index, value in enumerate(df_top[score_column]):
        plt.text(value, index, f'{value:.2f}', va='center', ha='left', color='black')
    plt.title(f'{title} ({percentile}° Percentile)')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Salva il grafico come PNG se richiesto
    if save_plot and filename:
        plt.savefig(f'{filename}_{percentile}.png')
        print(f'Grafico salvato come: {filename}_{percentile}.png')
        plt.close()
    else:
        plt.show()


def grafico_caratteristiche_selezionate(df, score_column, percentiles, top_n, title, xlabel, ylabel, save_plot=True,
                                        filename=None, csv_filename='feature_selezionate.csv'):
    df_features = pd.DataFrame()  # Creo dataframe vuoto
    df_bd = pd.DataFrame()
    for i, percentile in enumerate(percentiles):
        # Calcola la soglia in base al percentile
        soglia = np.percentile(df[score_column], percentile)

        # Filtra le feature che hanno un punteggio superiore alla soglia
        df_filtered = df[df[score_column] > soglia].copy()

        # Seleziona le prime 'top_n' feature dalla lista filtrata
        df_top = df_filtered.head(top_n)

        # Chiama la funzione per creare e salvare il barplot
        plot_barplot(df_top, score_column, title, xlabel, ylabel, percentile, filename, save_plot)

        # Aggiungo le caratteristiche selezionate al DataFrame
        df_features[f'Percentile_{percentiles[i]}'] = pd.Series(df_top['Feature'].values)


        df_bd[f'Percentile_{percentiles[i]}_Features'] = pd.Series(df_top['Feature'].values)
        df_bd[f'Percentile_{percentiles[i]}_Scores'] = pd.Series(df_top[score_column].values)

    # Salva le caratteristiche selezionate in un file CSV
    df_features.to_csv(csv_filename, index=False)
    print(f"File CSV salvato come '{csv_filename}'")

    return df_bd, df_features


# funzione per creare i boxplot
def funzione_boxplot(df, score_column, title, xlabel, ylabel, save_plot=True, filename=None):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[score_column])

    # Aggiungi titolo ed etichette degli assi
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Salva il grafico come PNG se richiesto
    if save_plot and filename:
        plt.savefig(filename)
        print(f'Grafico salvato come: {filename}')

    # Mostra il grafico
    plt.show()

    plt.close()


# Funzione per unire i DataFrame
def unisci_dataframes(mi, fi):
    # Unisci i DataFrame sulle colonne 'feature', mantenendo tutte le feature
    combined_df = pd.merge(mi, fi, on='Percentile_25_Features', how='inner').fillna(0)
    return combined_df


# Funzione per il barplot
def grafico_confronto_feature(df):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Definire la larghezza delle barre
    bar_width = 0.35
    index = np.arange(len(df))

    # Barre per MI
    ax.bar(index, df['Percentile_25_Scores_x'], bar_width, label='Mutual Information')

    # Barre per FI (spostate a destra di bar_width)
    ax.bar(index + bar_width, df['Percentile_25_Scores_y'], bar_width, label='Feature Importance')

    # Personalizzazione del grafico
    ax.set_xlabel('Feature')
    ax.set_ylabel('Score')
    ax.set_title('Confronto tra MI e FI per le feature')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(df['Percentile_25_Features'], rotation=90)
    ax.legend()

    plt.tight_layout()
    plt.show()

def smote_augmentation(x,y):
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X, Y = smote.fit_resample(x,y)
    return X,Y

def data_econder(x,y):
    encoder = ce.TargetEncoder(cols=x.select_dtypes(include=['object']).columns)
    X = encoder.fit_transform(x, y)
    return X

def MI_feature(x,y):
    mi_scores = mutual_info_classif(x,y, random_state=0)
    return mi_scores

def FI_feature(x,y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)
    model = RandomForestClassifier(random_state=0)
    model.fit(X_train, Y_train)

    importances = model.feature_importances_
    return importances



