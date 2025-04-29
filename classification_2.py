import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.tree import export_text, DecisionTreeClassifier
from sklearn import tree
import time
import gc
import tracemalloc

def read_file(filename):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(current_directory, filename)
    df = pd.read_csv(filepath)
    return df

def misura_tempo_modello(modello,X,Y, n_repeats=5): #mi calcolo la media
    tempi = []  #Lista per memorizzare i tempi di esecuzione validi
    while len(tempi) < n_repeats:
        start_time = time.process_time_ns()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
        modello.fit(X_train, Y_train)
        Y_pred = modello.predict(X_test)
        end_time = time.process_time_ns()
        tempo_esecuzione = end_time - start_time
        if tempo_esecuzione > 0:
            tempi.append(tempo_esecuzione)
    # Calcola la media dei tempi di esecuzione validi
    tempo_medio = np.mean(tempi)
    print(f"Tempo medio di training e testing: {tempo_medio:.1e} nanosecondi")

def misura_spazio_modello(modello, X, Y, n_repeats=5):
    spazi = []  # Lista per memorizzare lo spazio di memoria occupato in ciascuna ripetizione
    for _ in range(n_repeats):
        gc.collect()
        tracemalloc.start()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
        modello.fit(X_train, Y_train)
        Y_pred = modello.predict(X_test)
        spazio, picco = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        spazi.append(spazio)  # Aggiungi lo spazio alla lista

    # Calcola la media degli spazi
    spazio_medio = np.mean(spazi)

    # Stampa i risultati con meno cifre significative
    print("Lo spazio medio occupato reale è: {:.1e} byte".format(spazio_medio))
    return Y_pred, modello, Y_test

def random_forest(X, Y, num_alberi, depth1):
    rf = RandomForestClassifier(n_estimators=num_alberi, max_depth=depth1, criterion='gini', random_state=0, n_jobs=1)
    Y_pred, rf_add, Y_test= misura_spazio_modello(rf, X, Y)
    misura_tempo_modello(rf, X,Y)
    return rf_add, Y_pred, Y_test

def decision_tree(X, Y, depth2):
    # numero_chiamate = 0
    # numero_chiamate += 1
    albero = DecisionTreeClassifier(max_depth=depth2, criterion='gini', random_state=0)
    Y_pred, dt_add, Y_test = misura_spazio_modello(albero, X, Y)
    misura_tempo_modello(albero, X, Y)

    ''' # Visualizzazione dell'albero decisionale #non serve per le varie prove
    plt.figure(figsize=(10, 10))
    tree.plot_tree(albero, feature_names=X.columns, class_names=[str(cl)for cl in albero.classes_], filled=True)
    plt.title(f"Visualizzazione grafica dell'albero")
    plt.savefig(f"albero_{numero_chiamate}.png")
    plt.show()
    plt.close()

    # Visualizzare anche come testo
    albero_testuale = export_text(albero, feature_names=list(X.columns))
    print("Visualizzazione dell'Albero Decisionale (testuale):\n", albero_testuale)
    '''
    return dt_add, Y_pred, Y_test

def valutazione_modello(Y_test, Y_pred, modello, X, Y):
    cm = confusion_matrix(Y_test, Y_pred)
    print(f"Matrice di confusione:\n{cm}")
    # Estrazione valori dalla matrice di confusione (per un problema binario)
    TN = cm[0, 0]  # vero negativo, mi dice il posto della matrice in cui si trova
    FP = cm[0, 1]  # falso positivo
    FN = cm[1, 0]  # falso negativo
    TP = cm[1, 1]  # vero positivo

    # Calcolo delle metriche derivate
    accuracy = (TP+TN)/(TP+TN+FN+FP)
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    precision = precision_score(Y_test, Y_pred, average='weighted')
    recall = recall_score(Y_test, Y_pred,
                          average='weighted')  # average...serve per dar maggior peso alle classi che hanno piu elementi nel dataset
    f1 = f1_score(Y_test, Y_pred, average='weighted')

    cv_scores = cross_val_score(modello, X, Y, cv=5, n_jobs=1)
    print(f"Accuratezza: {accuracy * 100:.2f}%")
    print(f"Specificita': {specificity:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-score: {f1:.2f}")
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation score: {cv_scores.mean():.2f}")

    # Restituiamo tutte le metriche in un dizionario
    return {
        'accuracy': accuracy,
        'specificity': specificity,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'cross_val_mean': cv_scores.mean(),
        'cross_val_scores': cv_scores
    }


def vari_casi(X, Y, file_csv, num_alberi, depth1, depth2):
    caratteristiche_selezionate = pd.read_csv(file_csv)

    for percentile in caratteristiche_selezionate.columns:
        # 3 file csv in ogni file ho entrambi (alla volta 3 per fi 3 per mi )
        log_file = open(f'{file_csv}_{percentile}.txt', 'w')
        sys.stdout = log_file  # Reindirizziamo l'output al file di log
        # Otteniamo i nomi delle feature per questo percentile
        features_da_usare = caratteristiche_selezionate[percentile].dropna().tolist()

        print("valutazione RF con parametri fissati")
        modello_foresta, Y_pred_foresta, Y_test_foresta = random_forest(X[features_da_usare], Y, num_alberi, depth1)
        metriche_foresta = valutazione_modello(Y_test_foresta, Y_pred_foresta, modello_foresta,
                                               X[features_da_usare], Y)

        # Applica Decision Tree
        for misura in depth2:  # fisso due parametri cambia il terzo in base alle prove (in questo caso vario la depth2)
            print(f"\n valutazione DT_{misura}")
            modello_albero, Y_pred_albero, Y_test_albero = decision_tree(X[features_da_usare], Y, misura)
            metriche_albero = valutazione_modello(Y_test_albero, Y_pred_albero, modello_albero, X[features_da_usare], Y)

        log_file.close()
        sys.stdout = sys.__stdout__

def process_files(file1_path, file2_path, output_path):
    # Leggi i due file CSV
    df1 = pd.read_csv(file1_path)  # File 1: contiene diverse colonne come "Creat II", "Load Sistolico", etc.
    df2 = pd.read_csv(file2_path)  # File 2: contiene Percentile_25, Percentile_75, Percentile_90

    # Estrai i valori corrispondenti alla colonna "Percentile_90" di File 2
    percentiles_90 = df2['Percentile_90'].dropna().tolist()  # Lista dei valori di Percentile_90
    # Aggiungi la colonna 'DANNO EPI' (se esiste) dal File 1
    percentiles_90.append('DANNO EPI')

    # Crea un dizionario per mappare i valori dal file 1 per ogni colonna in percentiles_90
    data = {}

    for col in percentiles_90:
        # Controlla se la colonna esiste in df1
        if col in df1.columns:
            data[col] = df1[col].tolist()

    # Crea un nuovo DataFrame con le colonne corrispondenti a "Percentile_90"
    df_output = pd.DataFrame(data)

    # Salva il risultato in un nuovo file CSV
    df_output.to_csv(output_path, index=False)

    # Mostra il risultato finale (facoltativo)
    print(f"File salvato come {output_path}")
    print(df_output)




if __name__ == '__main__':
    #name = 'dataset_pre-processato.csv'
    name = 'dataset_pre-processato_2.csv'
    #name = 'dataset_pre-processato_comune.csv'
    #name = r'C:\Users\Laura Verde\Documents\Vanvitelli\TAF-D + tesi\Roberta-Giusy\DT_Tesi_Roberta_Giusy_ripulito_da_Pierluigi_modificato.xls'
    df = read_file(name)
    X = df.drop(columns=["DANNO EPI"])
    Y = df["DANNO EPI"]

    # prima prova
    vari_casi(X, Y, 'feature_mi.csv', 4, 1, [1, 2, 3])
    vari_casi(X, Y, 'feature_fi.csv', 4, 1, [1, 2, 3])

    log_file = open(f'results_Laura_S4.txt', 'w')
    sys.stdout = log_file  # Reindirizziamo l'output al file di log
    foresta_tot, Y1_pred_foresta, Y1_test_foresta = random_forest(X, Y, 4, 1)
    valutazione_foresta = valutazione_modello(Y1_test_foresta, Y1_pred_foresta, foresta_tot, X, Y)

    for misura in [1, 2, 3]:
        print(f"\n valutazione su tutte le feature cambiando lunghezza del DT_misura_{misura}")
        albero_tot, Y2_pred_albero, Y2_test_albero = decision_tree(X, Y, misura)
        valutazione_albero = valutazione_modello(Y2_test_albero, Y2_pred_albero, albero_tot, X, Y)
    log_file.close()
    sys.stdout = sys.__stdout__

    # Esegui la funzione
    process_files('dataset_pre-processato_2.csv', 'feature_mi.csv', 'MI_90.csv')


