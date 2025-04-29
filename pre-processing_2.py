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


def read_file(filename):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(current_directory, filename)
    df = pd.read_excel(filepath)
    return df


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


if __name__ == '__main__':
    print(sklearn.__version__)  # Assicurati che sia almeno 1.2+
    print(category_encoders.__version__)  # Dovrebbe essere almeno 2.6+
    pd.set_option('future.no_silent_downcasting', True)
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Apri un file di testo per l'output
    with open('output.txt', 'w') as f:
        sys.stdout = f  # Reindirizza stdout al file

        #name = r'C:\Users\User\PycharmProjects\pythonProject\Tesi\DT_Tesi_Roberta_Giusy_ripulito_da_Pierluigi.xls'
        name = r'C:\Users\Laura Verde\Documents\Vanvitelli\TAF-D + tesi\Roberta-Giusy\DT_Tesi_Roberta_Giusy_ripulito_da_Pierluigi.xls'
        df = read_file(name)
        df = df.fillna('nn')

        print("I numeri di righe e colonne del nostro data set sono rispettivamente: \n", df.shape)
        df.drop(df.columns[20:41], axis=1, inplace=True)  # Sono tutte le colonne riguardanti i dati della prima visita
        # che non ci interessano.

        # In giallo sono presenti i dati che è possibile ricavare dagli altri dati, ne elimino alcune.
        # Poiché ho gia eliminato in precedenza delle colonne, devo togliere 21 agli indici maggiori di 21
        # df.drop(df.columns[[16, 52-21, 94-21, 98-21, 99-21, 115-21, 116-21, 119-21, 121-21, 122-21]],axis=1,
        # inplace=True)
        df.drop(df.columns[[99 - 21, 100-21, 107-21, 106-21, 110-21]], axis=1,inplace=True)
        print("Successivamente, i numeri di righe e colonne del nostro data set sono rispettivamente: \n", df.shape)

        # Uso il metodo loc per eliminare le colonne con nomi duplicati mantenendo solo la prima occorrenza
        df = df.loc[:, ~df.columns.duplicated()]
        # Stampa il DataFrame aggiornato
        # print("\nDataFrame senza colonne con nomi duplicati:")
        # print(df)
        #df.replace("nn", np.nan, inplace=True)
        df.replace("nn", np.nan)
        df.infer_objects(copy=False)

        # Modifico le opzioni di visualizzazione per mostrare più righe
        pd.set_option('display.max_rows', None)
        # Ora df.dtypes mostrerà tutte le colonne
        print("Tipi colonne", df.dtypes)
        # Posso anche reimpostare le opzioni se necessario:
        # pd.reset_option('display.max_rows')

        df = detect_and_convert_to_boolean(df)
        # Modifico le opzioni di visualizzazione per mostrare più righe
        # pd.set_option('display.max_rows', None)
        print("I tipi errati sono stati corretti \n", df.dtypes)  # non so se è opportuno lasciare sta print
        pd.reset_option('display.max_rows')

        df = fill_missing_values(df)
        # Modifico le opzioni di visualizzazione per mostrare più righe
        # pd.set_option('display.max_columns', None)
        print("Riepilogo di tutte le colonne \n", df.describe(include="all"))



        # Mutual information
        # La mutual information ci permette di capire quale features in un dataset sono più rilevanti, rispetto a una
        # variabile target che nel nostro caso è Danno EPI
        X = df.drop(['DANNO EPI', '_1'], axis=1)  # Elimino anche '_1' perché non è una variabile utile a prevedere il
        # target
        Y = df['DANNO EPI']



        # converto le variabili categoriali di X in numeriche
        # X = pd.get_dummies(X)
        # Applica il target encoding alle variabili categoriali
        encoder = ce.TargetEncoder(cols=X.select_dtypes(include=['object']).columns)
        X = encoder.fit_transform(X, Y)  # Questo permetterà di mantenere lo stesso numero di colonne originale e
        # non ho piu bisogno di trasformare le variabili categoriali in molteplici colonne dummy.

        # salvo file pre-processato
        df_finale = pd.concat([X, Y], axis=1)
        df_finale.to_csv('dataset_pre-processato.csv', index=False)
        #print("Numero di righe e colonne df finale:", df_finale.shape)
        # Essendo il mio target categoriale (binario) vado a utilizzare mutual_info_classif

        ### SMOTE laura
        count_1 = (df['DANNO EPI'] == 1).sum()
        count_0 = (df['DANNO EPI'] == 0).sum()

        print(f"Valori 1 in Y: {count_1}")
        print(f"Valori  0 in Y: {count_0}")

        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X, Y = smote.fit_resample(X, Y)

        # Controlla le nuove classi
        print("Distribuzione originale:", Counter(Y))
        print("Distribuzione dopo SMOTE:", Counter(Y))

        df_finale = pd.concat([X, Y], axis=1)
        df_finale.to_csv('dataset_pre-processato_2.csv', index=False)

        ### SMOTE laura

        mi_scores = mutual_info_classif(X, Y, random_state=0)
        # La mutual information classification restituisce un array di punteggi, uno per ogni caratteristica.
        # Maggiore è il valore, maggiore è la dipendenza tra quella feature e la variabile target

        # Creo ora un dataframe che mi permette di visualizzare i punteggi
        mi_df = pd.DataFrame({'Feature': X.columns, 'MI Score': mi_scores})  # X.columns mi dà il nome delle col. di X
        # adesso ordino il dataframe creato sopra in base ai punteggi di mutual information
        mi_df = mi_df.sort_values(by='MI Score', ascending=False)

        # Utilizzo la funzione per il boxplot e ne creo uno per i punteggi MI
        funzione_boxplot(mi_df, 'MI Score', '', 'Mutual Information Score',
                         '', save_plot=True, filename='mi_boxplot.png')

        percentiles = [25, 75, 90]  # Lista variabile dei percentili.
        top_n = 30  # Numero delle top feature che seleziono, in genere infatti si seleziona l 20/30% delle variabili,
        # nel nostro caso dato che le variabili sono 102 ne ho considerate 30
        df_bd_mi, top_features_mi = grafico_caratteristiche_selezionate(mi_df, 'MI Score', percentiles, top_n,
                                   f'Top Feature - Mutual Information Scores ({percentiles}° Percentile)',
                                   'Mutual Information Score', 'Feature', save_plot=False, filename='MI_plot.png', csv_filename='feature_mi.csv')

        # Feature Importance with Random Forest

        # Standardizzazione delle caratteristiche
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Divido i dati in training e testing
        X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=0)

        # Divido i dati in training e testing
        # Il 30% dei dati lo uso per il testing e il restante 70% per il training
        # Utilizzo a validazione incrociata per valutare le prestazioni del modello

        model = RandomForestClassifier(random_state=0)
        model.fit(X_train, Y_train)

        # caratteristiche importanti
        importances = model.feature_importances_  # Una volta addestrato il modello, model.feature_importances_
        # restituisce un array che rappresenta l'importanza di ciascuna feature utilizzata nel modello.

        # Creazione di un DataFrame per visualizzare le caratteristiche importanti (con gli score come fatto per la MI)
        importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
        importance_df.sort_values(by='Importance', ascending=False, inplace=True)  # proprio come per la MI

        # Creo un boxplot per gli FI scores
        funzione_boxplot(importance_df, 'Importance', '',
                         'Feature Importance Score', '', save_plot=True, filename='fi_boxplot.png')

        percentiles = [25, 75, 90]
        top_n = 30
        df_bd_fi, top_features_fi = grafico_caratteristiche_selezionate(importance_df, 'Importance', percentiles, top_n,
                                                  f'Top {top_n} Feature - Random Forest ({percentiles}° Percentile)',
                                                  'Punteggio di Importanza', 'Feature', save_plot=False,
                                                  filename='RF_plot.png',csv_filename='feature_fi.csv')


        venn2((set(top_features_mi.iloc[:,0]), set(top_features_fi.iloc[:,0])), set_labels=('Mutual Information', 'Feature Importance'))
        #plt.title('Intersezione delle Feature Selezionate')
        plt.savefig("venn_diagram.png", format="png")
        plt.show()  # oss: con set converto in un insieme rimuovendo i duplicati

        # faccio questo diagramma di venn per evidenziare il fatto che alcune variabili sono in comune ma non tutte ora

        #per salvare le feature in comune -Laura
        # Calcolo dell'intersezione
        features_mi = set(top_features_mi.iloc[:, 0])
        features_fi = set(top_features_fi.iloc[:, 0])
        features_comuni = features_mi & features_fi  # Intersezione

        # Salvataggio su file txt
        with open("features_in_comune.txt", "w") as f:
            for feature in features_comuni:
                f.write(f"{feature}\n")

        # Salvataggio del dataset pre-processato con solo le feature in comune
        df_comuni = df_finale[list(features_comuni) + ['DANNO EPI']]
        df_comuni.to_csv('dataset_pre-processato_comune.csv', index=False)
        print("Dataset con solo le feature in comune salvato come 'dataset_pre-processato_comune.csv'")
        # per salvare le feature in comune -Laura

        scaler = MinMaxScaler()
        # Normalizza i valori MI e FI
        df_bd_mi.iloc[:, 1] = scaler.fit_transform(df_bd_mi.iloc[:, 1].values.reshape(-1, 1))
        df_bd_fi.iloc[:, 1] = scaler.fit_transform(df_bd_fi.iloc[:, 1].values.reshape(-1, 1))
        # faccio un barplot che mi associa alle caratteristiche in comune il rispettivo score scalate sopra
        combined_df = unisci_dataframes(df_bd_mi.iloc[:,:2], df_bd_fi.iloc[:,:2])
        pd.set_option('display.max_columns', None)
        print(combined_df)
        grafico_confronto_feature(combined_df)

        #aggiungere
        # Questo comando mi serve per porre fine alla scrittura nel file delle print
        sys.stdout = sys.__stdout__



