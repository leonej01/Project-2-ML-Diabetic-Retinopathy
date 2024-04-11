# Project: Machine Learning Diabetic-Retinopathy

# PROJECT #2

### Columbia School of Engineering AI Boot Camp


# **FANTASTIC 4**
---

<ul>(Team 4): 
  
<li>Jennifer Leone</li> 
<li>James O’Brien</li> 
<li>Osita Igwe</li> 
<li>Giancarlo Ocasio</li> 
<li>DoraMaria Abreu</li>
</ul>



# 04.4.24
---


## Objective:
Train an algorithim to predict whether an image contains signs of diabetic retinopathy or not.

### An executive summary or overview of the project and project goals (5 points).

This project focuses on developing and evaluating supervised machine learning algorithms for the automatic detection of diabetic retinopathy (DR) in retinal images. The goal is to train binary classification models that can accurately predict whether an image shows signs of DR or not.

The MESSIDOR (Methods to Evaluate Segmentation and Indexing Techniques in the field of Retinal Ophthalmology) dataset, consisting of 1,152 patient records, is being used to train and validate the models. This dataset provides a comprehensive set of features relevant to DR diagnosis.

To optimize the performance of the machine learning models, hyperparameter tuning and feature engineering were employed. Hyperparameter tuning involves systematically adjusting the model parameters to improve the prediction scores. Additionally, feature selection techniques are being applied to identify the most informative features for DR detection.

The insights gained from this exhaustive analysis will enable the development of robust models for automatic DR screening. By incorporating these models into clinical practice, doctors can more effectively identify patients who require further eyesight evaluation, leading to improved patient care and outcomes.

The Messidor project aims not only to develop accurate DR detection models but also to compare and evaluate different segmentation and indexing techniques in retinal ophthalmology. This comprehensive approach will contribute to advancing the field of automated eye disease diagnosis and ultimately enhance the efficiency and effectiveness of DR screening programs.

---

### Features
19 Total Features  
All features represent either a detected lesion, a descriptive feature of a anatomical part or an image-level descriptor. 
                            
- 0 :   Image Quality: 0 = Bad, 1 = Sufficient.
- 1 :   Pre-screening: 0 = No severe abnormality, 1 = Severe abnormality. 
- 2-7:  Microaneurysms (MA) detection at confidence levels 0.5 to 1
- 8-15: Exudates detection, normalized by ROI diameter.
- 16:   Euclidean distance between macula center and optic disc, normalized.
- 17:   Optic disc diameter.
- 18:   The binary result of the AM/FM-based classification. 
- 19:   Class label. 1 = contains signs of DR (Accumulative label for the Messidor classes 1, 2, 3), 0 = no signs of DR.

---

### Models Trained and Tested

* Logistic Regression, LogisticRegression()
* Support Vector Machine, SVC()
* K-Nearest Neighbors, KNeighborsClassifier()
* Decision Tree", DecisionTreeClassifier()
* Random Forest", RandomForestClassifier()
* Extremely Random Trees, ExtraTreesClassifier()
* Gradient Boosting, GradientBoostingClassifier()
* AdaBoost, AdaBoostClassifier()
* Naive Bayes, GaussianNB()  

---

## Data Exploration

### Data Source: 
Antal, Balint and Hajdu, Andras. (2014). Diabetic Retinopathy Debrecen. [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/329/diabetic+retinopathy+debrecen).


This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.

### Format: 
csv file

### Contents: 
1152 entries
20 columns including retina image quality, prescreening, evaluating and accuracy. 

### Data Clean-up: 
None needed. 

### dtypes: 
float64(10), int64(10). 
no string objects present to encode. 

[StandardScaler()](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) used after splitting data into test and train. 

Exploration:  Target is balanced: 

Class 1 (DR): 611 

Class 0 (No DR): 540

Main Starter File which calls the other file Model Utilities

---

### The approach that your group took in achieving the project goals.

- Collaborative planning and role assignment
- Research and methodological selection
- Iterative development and regular reviews
- Data driven decision making
- Evaluation and Interpretation

1. *Collaborative planning* and *role assignment*
2. *Research* and *methodological selection*
3. *Iterative development* and *regular reviews*
4. *Data driven decision making*
5. *Evaluation* and *Interpretation*

---

### The results/conclusions of the application or analysis:

Data-Driven Results: Accuracy was **NOT** perfect

Building Machine & Deep Learning Models,
Equip Doctors with better mothod for screening DR.
Creating a greater awareness of DR causes and symptoms.

---

# **CONCLUSION:**

## **Importance of the Project to the following topics**

- ## Early Detection: 
Early detection of Diabetic Retinopathy can significantly reduce the risk of severe vision loss. Machine learning models can assist in screening processes, making them more efficient and potentially more accurate than traditional methods.

- ## Scalability: 
Automated DR detection systems can scale to screen large populations, especially in resource-constrained environments where access to an ophthalmologists is limited.

- ## Clinical Decision Support: 
The project enhances clinical decision-making by providing a tool that aids in the prioritization of cases for review and intervention based on the risk of DR.

- ## Research Insights: 
Analysis of feature importance and model performance offers insights into the pathophysiology of DR, potentially guiding future research into its underlying mechanisms and treatments.

- ## Evolution in the Next Iteration
The project sets a foundation for leveraging advanced analytics in ophthalmology, with a clear path for iterative enhancements that can lead to more robust, clinically applicable models for Diabetic Retinopathy detection

- ## Deep Learning Approaches: 
Integrating deep learning models, particularly convolutional neural networks (CNNs), could leverage raw fundus images directly, potentially uncovering nuanced patterns not captured by engineered features.


- ## Ensemble Selection: 
After reviewing a scientific study that used the same dataset, we surmised that the lab used a backward ensemble search method.  It starts with all possible models and iteratively removes the weakest ones. We attempted to use a similar method, using only machine learning algorithms. However, our results were no better than those that were presented earlier. During the  next iterations, a similar approach using deep learning, might be a viable selection


- ## Broader Social Impact
Longitudinal Data: Incorporating longitudinal patient data could enable the development of models that predict DR progression over time, offering a dynamic tool for patient monitoring.


- ## Clinical Integration: 
Developing a pilot program for clinical validation and integration, involving feedback from healthcare professionals, can ensure the model's practical utility and acceptance.


- ## Explainability and Fairness: 
Implementing model explainability tools to interpret predictions and assess model fairness can ensure ethical application and trust among end-users.

---

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


### Additional questions that surfaced, what your group might research next if more time was available, or share a plan for future development.

If more time was available or for future development:
 our group could explore the following additional questions and research directions which could ***significantly enhance*** the project's ***impact*** and ***scalability***. These could aid in the creation of ***comprehensive*** and ***clinically applicable*** systems for the automated detection and management of Diabetic-Retinopathy.

### *Multiclass classification*: <ul>
<li>Extend the binary classification problem to multiclass classification, distinguishing between different stages of DR severity (e.g., no DR, mild, moderate, severe, proliferative DR). This would provide a more granular assessment of DR progression and help prioritize treatment strategies.</li></ul>

### *Interpretability and explainability:* <ul>
<li>Develop and integrate techniques for interpreting and explaining model predictions, such as saliency maps, attention mechanisms, or rule-based explanations. This would enhance the transparency and trustworthiness of the models, facilitating their adoption in clinical settings.</li></ul>

### *Integration with other clinical data:* <ul>
<li>Investigate the integration of additional clinical data, such as patient demographics, medical history, and systemic factors (e.g., diabetes duration, HbA1c levels), to improve the predictive power and robustness of the models.</li></ul>

### *Cost-effectiveness analysis:* <ul>
<li>Conduct a comprehensive cost-effectiveness analysis to assess the economic impact of implementing automated DR detection systems in various healthcare settings, considering factors such as screening costs, treatment costs, and quality-adjusted life years (QALYs) gained.</li></ul>

### *Human-AI collaboration:* <ul>
<li>Explore the design and evaluation of human-AI collaborative workflows, where the automated DR detection system works in tandem with human experts to optimize the screening process and decision-making. This could involve developing intuitive user interfaces and studying the impact on clinical workflows and outcomes.</li>

### *Continuous model updates:* <ul>
<li>Develop strategies for continuously updating and refining the models as new data becomes available, ensuring that the system adapts to evolving patient populations and captures the latest trends in DR manifestation.</li></ul>

### *Generalizability and external validation:* <ul>
<li>Assess the generalizability of the developed models by validating their performance on external datasets from diverse populations and imaging protocols. This would help establish the robustness and applicability of the models across different settings.</li></ul>

### *Integration with other eye diseases:* <ul>
<li>Expand the scope of the project to include the detection and management of other common eye diseases, such as glaucoma, age-related macular degeneration, and cataracts, leveraging the same framework and methodologies developed for DR detection.</li>
</ul>

### *Link for project on GitHub*
### [Machine Learning Diabetic-Retinopathy](https://github.com/leonej01/Project-2-ML-Diabetic-Retinopathy.git)

---
# Thank you!
---



