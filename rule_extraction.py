import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import _tree


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

