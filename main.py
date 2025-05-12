import os
import sys
import warnings
import category_encoders
import numpy as np
import pandas as pd
import sklearn
from sklearn.tree import export_text
import classification
import preprocessing
import rule_extraction
from packaging import version


def read_file(filename):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(current_directory, filename)
    dataframe = pd.read_excel(filepath)
    return dataframe

def sanity_check():
    sklearn_msg = f"scikit-learn >= 1.2 required, found {sklearn.__version__}"
    assert version.parse(sklearn.__version__) >= version.parse("1.2"), sklearn_msg
    category_encoders_msg = f"category_encoders >= 2.6 required, found {category_encoders.__version__}"
    assert version.parse(category_encoders.__version__) >= version.parse("2.6"), category_encoders_msg

def clean():
    extensions = ['txt', 'png', 'csv']
    for extension in extensions:
        removed_files = []
        for filename in os.listdir('./'):
            if filename.endswith(extension) and filename != "requirements.txt":
                file_path = os.path.join('./', filename)
                try:
                    os.remove(file_path)
                    removed_files.append(filename)
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")


def boot():
    pd.set_option('future.no_silent_downcasting', True)
    warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == '__main__':
    dataset_filename = sys.argv[1]
    sanity_check()
    boot()
    clean()
    with open('output.txt', 'w') as f:
        sys.stdout = f
        df = read_file(dataset_filename)
        df = df.fillna('nn')
        print("The number of rows and columns of our dataset are respectively: \n", df.shape)
        df.drop(df.columns[20:41], axis=1, inplace=True)
        # These are all columns concerning the first visit data, which are not relevant.
        # The highlighted data can be derived from other columns, so we remove some of them.
        # Since some columns were already removed earlier, we subtract 21 from indices > 21.
        df.drop(df.columns[[99 - 21, 100 - 21, 107 - 21, 106 - 21, 110 - 21]], axis=1, inplace=True)
        print("Afterwards, the number of rows and columns of our dataset are respectively: \n", df.shape)
        # Remove duplicate column names, keeping only the first occurrence
        df = df.loc[:, ~df.columns.duplicated()]
        df.replace("nn", np.nan)
        df = df.drop('Creat II', axis=1)
        df.infer_objects(copy=False)

        # Adjust display options to show more rows
        pd.set_option('display.max_rows', None)
        print("Column types", df.dtypes)

        df = preprocessing.detect_and_convert_to_boolean(df)
        # Uncomment to print data types after correction
        # print("Incorrect types have been corrected \n", df.dtypes)
        pd.reset_option('display.max_rows')
        df = preprocessing.fill_missing_values(df)
        # Uncomment to view summary of all columns
        # pd.set_option('display.max_columns', None)
        # print("Summary of all columns \n", df.describe(include="all"))

        X = df.drop(['DANNO EPI', '_1'], axis=1)
        Y = df['DANNO EPI']

        X = preprocessing.data_econder(X, Y)

        # Save the preprocessed dataset
        df_final = pd.concat([X, Y], axis=1)
        df_final.to_csv(r'preprocessed_dataset.csv',index=False)
        X, Y = preprocessing.smote_augmentation(X, Y)
        mi_scores = preprocessing.MI_feature(X, Y)
        # Mutual Information returns an array of scores, one per feature.
        # Higher scores indicate greater dependency with the target variable.

        # Create DataFrame to visualize MI scores
        mi_df = pd.DataFrame({'Feature': X.columns, 'MI Score': mi_scores})
        mi_df = mi_df.sort_values(by='MI Score', ascending=False)

        # Generate boxplot for MI scores
        preprocessing.funzione_boxplot(mi_df, 'MI Score', '', 'Mutual Information Score', '', save_plot=True,
                                       filename='mi_boxplot.png')

        percentiles = [25, 75, 90]  # List of percentiles
        top_n = 30  # Selecting top features (approx. 30% of total 102 features)

        df_bd_mi, top_features_mi = preprocessing.grafico_caratteristiche_selezionate(
            mi_df, 'MI Score', percentiles, top_n,
            f'Top Feature - Mutual Information Scores ({percentiles}° Percentile)',
            'Mutual Information Score', 'Feature',
            save_plot=False, filename='MI_plot.png',
            csv_filename='feature_mi.csv'
        )

        # Feature Importance via Random Forest
        importances = preprocessing.FI_feature(X, Y)
        importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
        importance_df.sort_values(by='Importance', ascending=False, inplace=True)

        preprocessing.funzione_boxplot(importance_df, 'Importance', '', 'Feature Importance Score', '',
                                       save_plot=True, filename='fi_boxplot.png')

        percentiles = [25, 75, 90]
        top_n = 30
        df_bd_fi, top_features_fi = preprocessing.grafico_caratteristiche_selezionate(
            importance_df, 'Importance', percentiles, top_n,
            f'Top {top_n} Feature - Random Forest ({percentiles}° Percentile)',
            'Importance Score', 'Feature',
            save_plot=False, filename='RF_plot.png',
            csv_filename='feature_fi.csv'
        )

        # End of file writing block
        sys.stdout = sys.__stdout__

        # Classification tasks using both feature selection methods
        classification.test_cases(X, Y, 'feature_mi.csv', 4, 1, [1, 2, 3])
        classification.test_cases(X, Y, 'feature_fi.csv', 4, 1, [1, 2, 3])

        log_file = open('results_classification.txt', 'w')
        sys.stdout = log_file
        foresta_tot, Y1_pred_foresta, Y1_test_foresta = classification.random_forest(X, Y, 4, 1)
        valutazione_foresta = classification.evaluate_model(Y1_test_foresta, Y1_pred_foresta, foresta_tot, X, Y)

        for misura in [1, 2, 3]:
            print(f"\n Evaluation using all features, changing tree depth (DT_misura_{misura})")
            albero_tot, Y2_pred_albero, Y2_test_albero = classification.decision_tree(X, Y, misura)
            valutazione_albero = classification.evaluate_model(Y2_test_albero, Y2_pred_albero, albero_tot, X, Y)

        log_file.close()
        sys.stdout = sys.__stdout__

        classification.process_files(
            r'preprocessed_dataset.csv',
            'feature_fi.csv', 'FI_90.csv')

        # Rule Extraction
        name = 'FI_90.csv'
        df = pd.read_csv(name)
        X = df.drop(columns=["DANNO EPI"])
        Y = df["DANNO EPI"]

        best_rf = rule_extraction.classification(X, Y)

        # Extract rules from individual decision trees
        rules = []
        for tree in best_rf.estimators_:
            r = export_text(tree, feature_names=list(X.columns), show_weights=True, max_depth=10)
            rules.append(r)

        for rule in rules[:4]:
            print(rule)

        feature_names = X.columns
        rules_list = []

        for tree in best_rf.estimators_:
            rules = rule_extraction.get_rules(tree, feature_names)
            rules_list.extend(rules)

        rules_df = pd.DataFrame(rules_list)
        rules_df.rename(columns={"rule": "Feature Rule"}, inplace=True)
        rules_df = rules_df[rules_df["class"] == 1].copy()
        rules_df.drop("class", axis=1, inplace=True)
        rules_df.rename(columns={"samples": "Total Samples"}, inplace=True)
        rules_df.rename(columns={"proba": "Fraud Probability"}, inplace=True)
        rules_df.reset_index(drop=True, inplace=True)

        print('Feature Rules:', "\n")
        pd.set_option('display.max_colwidth', None)
        for item in rules_df['Feature Rule'][0:20]:
            print(item, "\n")

        rules_df.to_csv('Random Forest Rules_mi.csv', index=False)
