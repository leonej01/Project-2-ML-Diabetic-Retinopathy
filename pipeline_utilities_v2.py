from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline, Pipeline
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def get_model_scores(X, y, model):
    """
    Returns the mean cross-validation score of a model.
    """
    scores = cross_val_score(model, X, y, cv=5)
    return scores.mean()

def get_model_scores_keep_na(X, y, model):
    """
    Returns the mean cross-validation score of a model.
    """
    scores = cross_val_score(model, X, y, cv=5)
    return scores.mean()
    

def preprocess_data(df):
    """
    Will drop null values and 
    split into training and testing sets. Uses price
    as the target column.
    """
    raw_num_df_rows = len(df)
    df = df.dropna()
    remaining_num_df_rows = len(df)
    percent_na = (
        (raw_num_df_rows - remaining_num_df_rows) / raw_num_df_rows * 100
    )
    print(f"Dropped {round(percent_na,2)}% rows")
    X = df.drop(columns='price')
    y = df['price'].values.reshape(-1, 1)
    return train_test_split(X, y)

def preprocess_data_keep_na(df):
    """
    Will split into training
    and testing sets. Uses price as the target column.
    """
    X = df.drop(columns='price')
    y = df['price'].values.reshape(-1, 1)
    return train_test_split(X, y)

def r2_adj(x, y, model):
    """
    Calculates adjusted r-squared values given an X variable, 
    predicted y values, and the model used for the predictions.
    """
    r2 = model.score(x,y)
    n_cols = x.shape[1]
    return 1 - (1 - r2) * (len(y) - 1) / (len(y) - n_cols - 1)

def check_metrics(X_test, y_test, model):
    # Use the pipeline to make predictions
    y_pred = model.predict(X_test)

    # Print out the MSE, r-squared, and adjusted r-squared values
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    print(f"R-squared: {r2_score(y_test, y_pred)}")
    print(f"Adjusted R-squared: {r2_adj(X_test, y_test, model)}")

    return r2_adj(X_test, y_test, model)

def get_best_pipeline(pipeline, pipeline2, df):
    """
    Accepts two pipelines and data.
    Uses two different preprocessing functions to 
    split the data for training the different 
    pipelines, then evaluates which pipeline performs
    best.
    """
    # Apply the preprocess_data step
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Fit the first pipeline
    pipeline.fit(X_train, y_train)

    print("Testing dropped NAs")
    # Print out the MSE, r-squared, and adjusted r-squared values
    # and collect the adjusted r-squared for the first pipeline
    p1_adj_r2 = check_metrics(X_test, y_test, pipeline)

    # Apply the preprocess_data_keep_na step
    X_train, X_test, y_train, y_test = preprocess_data_keep_na(df)

    # Fit the second pipeline
    pipeline2.fit(X_train, y_train)

    print("Testing no dropped data")
    # Print out the MSE, r-squared, and adjusted r-squared values
    # and collect the adjusted r-squared for the second pipeline
    p2_adj_r2 = check_metrics(X_test, y_test, pipeline2)

    # Compare the adjusted r-squared for each pipeline and 
    # return the best model
    if p2_adj_r2 > p1_adj_r2:
        print("Returning no dropped data pipeline")
        return pipeline2
    else:
        print("Returning dropped NAs pipeline")
        return pipeline

def model_generator(X_test, y_test, X_train, y_train):
    """
    Defines a series of steps that will preprocess data,
    split data, and train multiple models. It will return the trained model
    and print the mean squared error, r-squared, and adjusted
    r-squared scores.
    """

    # Define Models
    models = [
            ("Logistic Regression", LogisticRegression()),
            ("Support Vector Machine", SVC()),
            ("K-Nearest Neighbors", KNeighborsClassifier()),
            ("Decision Tree", DecisionTreeClassifier()),
            ("Random Forest", RandomForestClassifier()),
            ("Extremely Random Trees", ExtraTreesClassifier()),
            ("Gradient Boosting", GradientBoostingClassifier()),
            ("AdaBoost", AdaBoostClassifier()),
            ("Naive Bayes", GaussianNB())]
    
    # Create a list of steps for a pipeline that will one hot encode and scale data
    # Each step should be a tuple with a name and a function
    
    steps = [("Scale", StandardScaler(with_mean=False))] 
    
    for name, model in models:
        steps.append((name, model))
        
    # Create a pipeline object
    pipeline = Pipeline(steps)
    
    # Fit the pipeline to the training data
    pipeline.fit(X_train, y_train)
 
    # Evaluate the pipeline on the test data and print the accuracy and confusion matrix
    name = pipeline.steps[-1][0]
    print(f"Evaluating {name}...")
    accuracy = pipeline.score(X_test, y_test)
    print(f"{name} - Accuracy: {accuracy:.4f}")

    confusion_matrix_results = confusion_matrix(y_test, pipeline.predict(X_test))
    print(f"{name} - Confusion Matrix:\n{confusion_matrix_results}")

    # Return the trained model
    return pipeline, accuracy, confusion_matrix_results
