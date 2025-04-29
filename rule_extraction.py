# Importing necessary libraries
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import _tree, export_text
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

def read_file(filename):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(current_directory, filename)
    df = pd.read_csv(filepath)
    return df

def classification (X,Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    best_rf = RandomForestClassifier(n_estimators=4, max_depth=2, criterion='gini', random_state=0, n_jobs=1)
    best_rf.fit(X_train, Y_train)

    return best_rf


# Defining function to extract rules from a decision tree
def get_rules(tree, feature_names):
    tree_ = tree.tree_  # Getting the underlying tree structure
    # Creating a list of feature names, replacing undefined features with a placeholder string
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []  # List to store all the paths (rules) in the tree
    path = []  # Temporary list to store the current path

    # Recursive function to traverse the tree and collect rules
    def recurse(node, path, paths):
        # If the node is not a leaf node
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 6)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 6)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            # If the node is a leaf node, append the current path to paths
            path += [(tree_.value[node], np.sum(tree_.value[node]))]
            paths += [path]

    # Starting the recursive function from the root node
    recurse(0, path, paths)

    # Sorting the paths based on the number of samples
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]

    rules = []  # List to store the rules in a formatted way
    for path in paths:
        rule = "if "

        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)

        classes = path[-1][0][0]
        l = np.argmax(classes)
        class_label = l
        proba = np.round(100.0 * classes[l] / np.sum(classes), 2)
        samples = path[-1][1]

        # Appending the formatted rule to the rules list
        rules.append({
            'rule': rule,
            'class': class_label,
            'samples': samples,
            'proba': proba
        })

    return rules



if __name__ == '__main__':
    name = 'FI_90.csv'
    df = read_file(name)
    X = df.drop(columns=["DANNO EPI"])
    Y = df["DANNO EPI"]

    best_rf=classification(X,Y)

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
        rules = get_rules(tree, feature_names)
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
