from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import export_text
from sklearn.model_selection import cross_val_score, KFold

def run_decision_tree(data, labels, feature_names, max_depth=2):
    clf = DecisionTreeClassifier(max_depth=max_depth, criterion="gini")
    # Fit the decision tree to the entire dataset
    clf.fit(data, labels)
    print("Decision Tree:")
    tree_text = export_text(clf, feature_names=feature_names)
    print(tree_text)
    # Make predictions on the entire dataset
    predictions = clf.predict(data)
    rules_applied = clf.apply(data)
    for sample, rule in enumerate(rules_applied):
        print(f"Sample {sample + 1} is classified using rule {rule} in the decision tree.")
    # Calculate and print performance metrics for the entire dataset
    accuracy = accuracy_score(labels, predictions)
    confusion = confusion_matrix(labels, predictions)
    classification_rep = classification_report(labels, predictions)
    print("Performance Metrics for Decision Tree:")
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{confusion}")
    print(f"Classification Report:\n{classification_rep}")

    return clf

def run_decision_tree_cv(data, labels, feature_names, max_depth=None):
    clf = DecisionTreeClassifier(max_depth=max_depth, criterion="entropy")
    
    # Set up 8-fold cross-validation
    cv = KFold(n_splits=8)
    tree_list = []  # List to store decision trees
    
    for train_index, test_index in cv.split(data):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        # Fit the decision tree to the training data
        clf.fit(X_train, y_train)
        tree_list.append(clf)  # Store the trained decision tree
        
        # Make predictions on the test data
        predictions = clf.predict(X_test)
        
        # Calculate and print performance metrics for each fold
        accuracy = accuracy_score(y_test, predictions)
        confusion = confusion_matrix(y_test, predictions)
        classification_rep = classification_report(y_test, predictions)
        
        print(f"Fold {len(tree_list)} Decision Tree:")
        tree_text = export_text(clf, feature_names=feature_names)
        print(tree_text)
        print("\nPerformance Metrics for Fold", len(tree_list))
        print(f"Accuracy: {accuracy}")
        print(f"Confusion Matrix:\n{confusion}")
        print(f"Classification Report:\n{classification_rep}")
    
    # Print the mean accuracy across all folds
    scores = cross_val_score(clf, data, labels, cv=cv)
    print("Cross-Validation Scores:")
    print(scores)
    print(f"Mean Accuracy: {scores.mean()}")
    
    # Fit the final model to the entire dataset
    clf.fit(data, labels)
    
    return clf, tree_list