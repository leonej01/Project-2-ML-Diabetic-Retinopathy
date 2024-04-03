# model_utilities.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns


def load_data(url):
    """Load data from a URL into a pandas DataFrame."""
    return pd.read_csv(url)
   
def preprocess_data(df, target_column):
    """Preprocess data by splitting into features and target, and then train-test split."""
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the test set."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    print("Classification Report:\n", classification_report(y_test, predictions))
    return accuracy, confusion_matrix(y_test, predictions), classification_report(y_test, predictions)
    

def plot_model_scores(model_scores):
    # Sort model_scores by the score, which is the second item in each tuple
    sorted_scores = sorted(model_scores, key=lambda x: x[1])
    
    # Unpack model names and their scores into separate lists
    names, scores = zip(*sorted_scores)
    
    # Create a bar chart
    plt.figure(figsize=(10, 8))  # Set the figure size for better readability
    bars = plt.bar(names, scores, color='skyblue')  # Plot the sorted scores with model names as x-axis labels
    
    plt.xlabel('Model')  # Set x-axis label
    plt.ylabel('Accuracy Score')  # Set y-axis label
    plt.title('Model Accuracy Comparison')  # Set the title of the chart
    plt.xticks(rotation=45, ha='right')  # Rotate the x-axis labels for better readability
    
    # Add a data label on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, round(yval, 4), ha='center', va='bottom')
    
    # Display the plot
    plt.tight_layout()  # Adjust layout to not cut off labels
    plt.show()
    

def apply_grid_search(models_with_params, X_train, y_train, cv=5, scoring='accuracy'):
    """
    Apply GridSearchCV to a list of models with their parameter grids.

    :param models_with_params: A list of tuples, each containing a model and its parameter grid.
    :param X_train: Training features.
    :param y_train: Training labels.
    :param cv: Number of cross-validation folds.
    :param scoring: Scoring method to evaluate the predictions on the test set.
    :return: A list of dictionaries with best parameters and scores for each model.
    """
    results = []

    for model, params in models_with_params:
        grid_search = GridSearchCV(estimator=model, param_grid=params, cv=cv, scoring=scoring, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        results.append({
            'model': model.__class__.__name__,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        })

    return results


def apply_random_search(models_with_params, X_train, y_train, cv=5, scoring='accuracy', n_iter=100, random_state=None):
    """
    Apply RandomizedSearchCV to a list of models with their parameter grids.

    :param models_with_params: A list of tuples, each containing a model and its parameter grid.
    :param X_train: Training features.
    :param y_train: Training labels.
    :param cv: Number of cross-validation folds.
    :param scoring: Scoring method to evaluate the predictions on the test set.
    :param n_iter: Number of parameter settings that are sampled. n_iter trades off runtime vs quality of the solution.
    :param random_state: Pass an int for reproducible output across multiple function calls.
    :return: A list of dictionaries with best parameters and scores for each model.
    """
    results = []

    for model, params in models_with_params:
        random_search = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=n_iter, cv=cv, scoring=scoring, random_state=random_state, n_jobs=-1)
        random_search.fit(X_train, y_train)
        
        results.append({
            'model': model.__class__.__name__,
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_
        })

    return results



def plot_confusion_matrix(y_true, y_pred, class_names, figsize=(10, 7), cmap='Blues'):
    """
    Plots a confusion matrix using Seaborn's heatmap.

    :param y_true: Array of true target values
    :param y_pred: Array of predicted values by the model
    :param class_names: List of class names for the target variable
    :param figsize: Tuple representing figure size (width, height) in inches
    :param cmap: Colormap recognized by matplotlib
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize the confusion matrix

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, cmap=cmap, xticklabels=class_names, yticklabels=class_names)
    
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()
