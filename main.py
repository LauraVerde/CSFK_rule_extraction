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

from sklearn.tree import export_text

import preprocessing
import classification


import sklearn
import category_encoders
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import rule_extraction


def read_file(filename):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(current_directory, filename)
    df = pd.read_excel(filepath)
    return df

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(sklearn.__version__)  # Assicurati che sia almeno 1.2+
    print(category_encoders.__version__)  # Dovrebbe essere almeno 2.6+
    pd.set_option('future.no_silent_downcasting', True)
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Apri un file di testo per l'output
    with open('output.txt', 'w') as f:
        sys.stdout = f  # Reindirizza stdout al file

        # name = r'C:\Users\User\PycharmProjects\pythonProject\Tesi\DT_Tesi_Roberta_Giusy_ripulito_da_Pierluigi.xls'
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
        df.drop(df.columns[[99 - 21, 100 - 21, 107 - 21, 106 - 21, 110 - 21]], axis=1, inplace=True)
        print("Successivamente, i numeri di righe e colonne del nostro data set sono rispettivamente: \n", df.shape)

        # Uso il metodo loc per eliminare le colonne con nomi duplicati mantenendo solo la prima occorrenza
        df = df.loc[:, ~df.columns.duplicated()]
        df.replace("nn", np.nan)
        df = df.drop('Creat II', axis=1)
        df.infer_objects(copy=False)

        # Modifico le opzioni di visualizzazione per mostrare più righe
        pd.set_option('display.max_rows', None)
        print("Tipi colonne", df.dtypes)


        df = preprocessing.detect_and_convert_to_boolean(df)
        # Modifico le opzioni di visualizzazione per mostrare più righe
        #print("I tipi errati sono stati corretti \n", df.dtypes)  # non so se è opportuno lasciare sta print
        pd.reset_option('display.max_rows')

        df = preprocessing.fill_missing_values(df)
        # Modifico le opzioni di visualizzazione per mostrare più righe
        # pd.set_option('display.max_columns', None)
        #print("Riepilogo di tutte le colonne \n", df.describe(include="all"))



        X = df.drop(['DANNO EPI', '_1'], axis=1)
        Y = df['DANNO EPI']


        X=preprocessing.data_econder(X,Y)

        # salvo file pre-processato
        df_finale = pd.concat([X, Y], axis=1)
        df_finale.to_csv(r'C:\Users\Laura Verde\Documents\Vanvitelli\Articoli\KES 2025\code\dataset_pre-processato.csv', index=False)


        X, Y = preprocessing.smote_augmentation(X, Y)

        mi_scores = preprocessing.MI_feature(X,Y)
        # La mutual information classification restituisce un array di punteggi, uno per ogni caratteristica.
        # Maggiore è il valore, maggiore è la dipendenza tra quella feature e la variabile target

        # Creo ora un dataframe che mi permette di visualizzare i punteggi
        mi_df = pd.DataFrame({'Feature': X.columns, 'MI Score': mi_scores})  # X.columns mi dà il nome delle col. di X
        # adesso ordino il dataframe creato sopra in base ai punteggi di mutual information
        mi_df = mi_df.sort_values(by='MI Score', ascending=False)

        # Utilizzo la funzione per il boxplot e ne creo uno per i punteggi MI
        preprocessing.funzione_boxplot(mi_df, 'MI Score', '', 'Mutual Information Score',
                         '', save_plot=True, filename='mi_boxplot.png')

        percentiles = [25, 75, 90]  # Lista variabile dei percentili.
        top_n = 30  # Numero delle top feature che seleziono, in genere infatti si seleziona l 20/30% delle variabili,
        # nel nostro caso dato che le variabili sono 102 ne ho considerate 30
        df_bd_mi, top_features_mi = preprocessing.grafico_caratteristiche_selezionate(mi_df, 'MI Score', percentiles, top_n,
                                                                        f'Top Feature - Mutual Information Scores ({percentiles}° Percentile)',
                                                                        'Mutual Information Score', 'Feature',
                                                                        save_plot=False, filename='MI_plot.png',
                                                                        csv_filename='feature_mi.csv')

        # Feature Importance with Random Forest
        importances=preprocessing.FI_feature(X,Y)
        # Creazione di un DataFrame per visualizzare le caratteristiche importanti (con gli score come fatto per la MI)
        importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
        importance_df.sort_values(by='Importance', ascending=False, inplace=True)  # proprio come per la MI

        # Creo un boxplot per gli FI scores
        preprocessing.funzione_boxplot(importance_df, 'Importance', '',
                         'Feature Importance Score', '', save_plot=True, filename='fi_boxplot.png')

        percentiles = [25, 75, 90]
        top_n = 30
        df_bd_fi, top_features_fi = preprocessing.grafico_caratteristiche_selezionate(importance_df, 'Importance', percentiles, top_n,
                                                                        f'Top {top_n} Feature - Random Forest ({percentiles}° Percentile)',
                                                                        'Punteggio di Importanza', 'Feature',
                                                                        save_plot=False,
                                                                        filename='RF_plot.png',
                                                                        csv_filename='feature_fi.csv')

        # aggiungere
        # Questo comando mi serve per porre fine alla scrittura nel file delle print
        sys.stdout = sys.__stdout__



        # CLASSIFICATION
        classification.vari_casi(X, Y, 'feature_mi.csv', 4, 1, [1, 2, 3])
        classification.vari_casi(X, Y, 'feature_fi.csv', 4, 1, [1, 2, 3])

        log_file = open(f'results_classification.txt', 'w')
        sys.stdout = log_file  # Reindirizziamo l'output al file di log
        foresta_tot, Y1_pred_foresta, Y1_test_foresta = classification.random_forest(X, Y, 4, 1)
        valutazione_foresta = classification.valutazione_modello(Y1_test_foresta, Y1_pred_foresta, foresta_tot, X, Y)

        for misura in [1, 2, 3]:
            print(f"\n valutazione su tutte le feature cambiando lunghezza del DT_misura_{misura}")
            albero_tot, Y2_pred_albero, Y2_test_albero = classification.decision_tree(X, Y, misura)
            valutazione_albero = classification.valutazione_modello(Y2_test_albero, Y2_pred_albero, albero_tot, X, Y)
        log_file.close()
        sys.stdout = sys.__stdout__

        #classification.process_files(r'C:\Users\Laura Verde\Documents\Vanvitelli\Articoli\KES 2025\code\dataset_pre-processato.csv', 'feature_mi.csv', 'MI_90.csv')
        classification.process_files(
            r'C:\Users\Laura Verde\Documents\Vanvitelli\Articoli\KES 2025\code\dataset_pre-processato.csv',
            'feature_fi.csv', 'FI_90.csv')

        #Rule-extraction
        name = 'FI_90.csv'
        df = pd.read_csv(name)
        X = df.drop(columns=["DANNO EPI"])
        Y = df["DANNO EPI"]

        best_rf = rule_extraction.classification(X, Y)

        # Initializing an empty list to store rules from decision trees
        rules = []

        # Iterating through each decision tree in the random forest
        for tree in best_rf.estimators_:
            # Exporting the structure of the decision tree as text and appending it to the rules list
            r = export_text(tree, feature_names=list(X.columns), show_weights=True, max_depth=10)
            rules.append(r)

        # Printing rules of the first 4 decision trees
        for rule in rules[:4]:
            print(rule)

        # Getting feature names from the DataFrame df_train, excluding the target variable column
        feature_names = X.columns

        # Creating an empty list to store the rules from each tree
        rules_list = []

        # Iterating through each decision tree in the random forest to collect rules
        for tree in best_rf.estimators_:
            rules = rule_extraction.get_rules(tree, feature_names)
            rules_list.extend(rules)

        # Converting the list of dictionaries into a DataFrame
        rules_df = pd.DataFrame(rules_list)

        # Renaming columns for better interpretation
        rules_df.rename(columns={"rule": "Feature Rule"}, inplace=True)
        rules_df = rules_df[rules_df["class"] == 1].copy()
        rules_df.drop("class", axis=1, inplace=True)
        rules_df.rename(columns={"samples": "Total Samples"}, inplace=True)
        rules_df.rename(columns={"proba": "Fraud Probability"}, inplace=True)
        rules_df.reset_index(drop=True, inplace=True)

        # Displaying the first 5 rows of the rules DataFrame
        rules_df.head()

        # Print best rules
        print('Feature Rules:', "\n")
        pd.set_option('display.max_colwidth', None)
        for item in rules_df['Feature Rule'][0:20]:
            print(item, "\n")

        # Export best rules
        rules_df.to_csv('Random Forest Rules_mi.csv', index=False)
