# Project: Machine Learning Diabetic-Retinopathy

# PROJECT #2

### Columbia School of Engineering AI Boot Camp


### FANTASTIC 4
(Team 4): 
Jennifer Leone, 
James O’Brien, 
Osita Igwe, 
Giancarlo Ocasio, 
DoraMaria Abreu

### 04.4.24

Objective: Train an algorithim to predict whether an image contains signs of diabetic retinopathy or not.


### An executive summary or overview of the project and project goals (5 points).

This project focuses on developing and evaluating supervised machine learning algorithms for the automatic detection of diabetic retinopathy (DR) in retinal images. The goal is to train binary classification models that can accurately predict whether an image shows signs of DR or not.

The MESSIDOR (Methods to Evaluate Segmentation and Indexing Techniques in the field of Retinal Ophthalmology) dataset, consisting of 1,152 patient records, is being used to train and validate the models. This dataset provides a comprehensive set of features relevant to DR diagnosis.

To optimize the performance of the machine learning models, hyperparameter tuning and feature engineering were employed. Hyperparameter tuning involves systematically adjusting the model parameters to improve the prediction scores. Additionally, feature selection techniques are being applied to identify the most informative features for DR detection.

The insights gained from this exhaustive analysis will enable the development of robust models for automatic DR screening. By incorporating these models into clinical practice, doctors can more effectively identify patients who require further eyesight evaluation, leading to improved patient care and outcomes.

The Messidor project aims not only to develop accurate DR detection models but also to compare and evaluate different segmentation and indexing techniques in retinal ophthalmology. This comprehensive approach will contribute to advancing the field of automated eye disease diagnosis and ultimately enhance the efficiency and effectiveness of DR screening programs.



### An overview of the data collection, cleanup, and exploration processes. Include a description of how you evaluated the trained model(s) using testing data. (5 points)

Data Source: Antal,Balint and Hajdu,Andras. (2014). Diabetic Retinopathy Debrecen. UCI Machine Learning Repository. https://doi.org/10.24432/C5XP4P.
This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.

### Data Dictionary


### Preprocessing

Logistic Regression, LogisticRegression()
Support Vector Machine, SVC()
K-Nearest Neighbors, KNeighborsClassifier()
Decision Tree", DecisionTreeClassifier()
Random Forest", RandomForestClassifier()
Extremely Random Trees, ExtraTreesClassifier()
Gradient Boosting, GradientBoostingClassifier()
AdaBoost, AdaBoostClassifier()
Naive Bayes, GaussianNB()


### Data Exploration

Data Source: Antal,Balint and Hajdu,Andras. (2014). Diabetic Retinopathy Debrecen. UCI Machine Learning Repository. https://doi.org/10.24432/C5XP4P. This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.
Format: csv file
Contents: 1152 entries, 20 columns including retina image quality, prescreening, evaluating and accuracy. 
Data Clean-up: None needed. dtypes: float64(10), int64(10). no string objects present to encode. StandardScalar() used after splitting data into train, test. 
Exploration:  Target is balanced: 
Class 1 (DR): 611 
Class 0 (No DR): 540
Main Starter File and which calls the other file Model Utilities


### The approach that your group took in achieving the project goals (5 points).

Collaborative planning and role assignment
Research and methodological selection
Iterative development and regular reviews
Data driven decision making
Evaluation and Interpretation



### The results/conclusions of the application or analysis:


### Conclusion:


Importance of the Project
Early Detection: Early detection of Diabetic Retinopathy can significantly reduce the risk of severe vision loss. Machine learning models can assist in screening processes, making them more efficient and potentially more accurate than traditional methods.
Scalability: Automated DR detection systems can scale to screen large populations, especially in resource-constrained environments where access to an ophthalmologists is limited.
Clinical Decision Support: The project enhances clinical decision-making by providing a tool that aids in the prioritization of cases for review and intervention based on the risk of DR.
Research Insights: Analysis of feature importance and model performance offers insights into the pathophysiology of DR, potentially guiding future research into its underlying mechanisms and treatments.


Evolution in the Next Iteration
The project sets a foundation for leveraging advanced analytics in ophthalmology, with a clear path for iterative enhancements that can lead to more robust, clinically applicable models for Diabetic Retinopathy detection
Deep Learning Approaches: Integrating deep learning models, particularly convolutional neural networks (CNNs), could leverage raw fundus images directly, potentially uncovering nuanced patterns not captured by engineered features.


Ensemble Selection: After reviewing a scientific study that used the same dataset, we surmised that the lab used a backward ensemble search method.  It starts with all possible models and iteratively removes the weakest ones. We attempted to use a similar method, using only machine learning algorithms. However, our results were no better than those that were presented earlier. During the  next iterations, a similar approach using deep learning, might be a viable selection

Broader Social Impact
Longitudinal Data: Incorporating longitudinal patient data could enable the development of models that predict DR progression over time, offering a dynamic tool for patient monitoring.
Clinical Integration: Developing a pilot program for clinical validation and integration, involving feedback from healthcare professionals, can ensure the model's practical utility and acceptance.
Explainability and Fairness: Implementing model explainability tools to interpret predictions and assess model fairness can ensure ethical application and trust among end-users.




### Additional questions that surfaced, what your group might research next if more time was available, or share a plan for future development (5 points).

### Link for project on GitHub
https://github.com/leonej01/Project-2-ML-Diabetic-Retinopathy.git


### Further Analysis
