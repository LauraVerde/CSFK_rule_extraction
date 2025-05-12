import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.tree import export_text, DecisionTreeClassifier
import time
import gc
import tracemalloc

def measure_model_time(model, X, Y, n_repeats=5):
    times = []  # List to store valid execution times
    while len(times) < n_repeats:
        start_time = time.process_time_ns()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        end_time = time.process_time_ns()
        elapsed_time = end_time - start_time
        if elapsed_time > 0:
            times.append(elapsed_time)
    average_time = np.mean(times)
    print(f"Average training and testing time: {average_time:.1e} nanoseconds")

def measure_model_memory(model, X, Y, n_repeats=5):
    memories = []  # List to store memory usage in each repeat
    for _ in range(n_repeats):
        gc.collect()
        tracemalloc.start()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        memory, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memories.append(memory)
    average_memory = np.mean(memories)
    print("Average actual memory used: {:.1e} bytes".format(average_memory))
    return Y_pred, model, Y_test

def random_forest(X, Y, num_trees, max_depth):
    rf = RandomForestClassifier(n_estimators=num_trees, max_depth=max_depth, criterion='gini', random_state=0, n_jobs=1)
    Y_pred, trained_rf, Y_test = measure_model_memory(rf, X, Y)
    measure_model_time(rf, X, Y)
    return trained_rf, Y_pred, Y_test

def decision_tree(X, Y, max_depth):
    tree_model = DecisionTreeClassifier(max_depth=max_depth, criterion='gini', random_state=0)
    Y_pred, trained_tree, Y_test = measure_model_memory(tree_model, X, Y)
    measure_model_time(tree_model, X, Y)

    '''
    # Decision tree visualization (not used in tests)
    plt.figure(figsize=(10, 10))
    tree.plot_tree(tree_model, feature_names=X.columns, class_names=[str(cl) for cl in tree_model.classes_], filled=True)
    plt.title("Graphical representation of the tree")
    plt.savefig(f"tree_visualization.png")
    plt.show()
    plt.close()

    # Textual representation of the tree
    textual_tree = export_text(tree_model, feature_names=list(X.columns))
    print("Textual Decision Tree Visualization:\n", textual_tree)
    '''
    return trained_tree, Y_pred, Y_test

def evaluate_model(Y_test, Y_pred, model, X, Y):
    cm = confusion_matrix(Y_test, Y_pred)
    print(f"Confusion Matrix:\n{cm}")
    TN, FP = cm[0, 0], cm[0, 1]
    FN, TP = cm[1, 0], cm[1, 1]

    accuracy = (TP + TN) / (TP + TN + FN + FP)
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    precision = precision_score(Y_test, Y_pred, average='weighted')
    recall = recall_score(Y_test, Y_pred, average='weighted')
    f1 = f1_score(Y_test, Y_pred, average='weighted')

    cv_scores = cross_val_score(model, X, Y, cv=5, n_jobs=1)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Specificity: {specificity:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-score: {f1:.2f}")
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation score: {cv_scores.mean():.2f}")

    return {
        'accuracy': accuracy,
        'specificity': specificity,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'cross_val_mean': cv_scores.mean(),
        'cross_val_scores': cv_scores
    }

def test_cases(X, Y, csv_file, num_trees, rf_depth, dt_depths):
    selected_features = pd.read_csv(csv_file)

    for percentile in selected_features.columns:
        log_file = open(f'{csv_file}_{percentile}.txt', 'w')
        sys.stdout = log_file  # Redirect output to log file
        features = selected_features[percentile].dropna().tolist()

        print("Evaluation of RF with fixed parameters")
        rf_model, Y_pred_rf, Y_test_rf = random_forest(X[features], Y, num_trees, rf_depth)
        rf_metrics = evaluate_model(Y_test_rf, Y_pred_rf, rf_model, X[features], Y)

        for depth in dt_depths:
            print(f"\nEvaluation DT_{depth}")
            dt_model, Y_pred_dt, Y_test_dt = decision_tree(X[features], Y, depth)
            dt_metrics = evaluate_model(Y_test_dt, Y_pred_dt, dt_model, X[features], Y)

        log_file.close()
        sys.stdout = sys.__stdout__

def process_files(file1_path, file2_path, output_path):
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    percentile_90 = df2['Percentile_90'].dropna().tolist()
    percentile_90.append('DAMAGE EPI')

    data = {}
    for col in percentile_90:
        if col in df1.columns:
            data[col] = df1[col].tolist()

    df_output = pd.DataFrame(data)
    df_output.to_csv(output_path, index=False)
    print(f"File saved as {output_path}")
    print(df_output)