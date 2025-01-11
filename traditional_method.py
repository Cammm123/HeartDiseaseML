# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 07:01:39 2024

@author: kani2
"""

#%%
"""
THE BASICS
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

# sklearn models (I left out SVMs -> The most basic ML algorithms)
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

# useful sklearn functions
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Some metrics to import
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, fbeta_score, roc_curve, roc_auc_score

# xgboost models
from xgboost import XGBRegressor
from xgboost import XGBClassifier

# unbalanced data problem
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import TomekLinks, RandomUnderSampler
from imblearn.combine import SMOTETomek
cvd_data = pd.read_csv('C:\\Users\\kani2\\OneDrive\\Desktop\\Academics\\CodingStuff\\DataSets\\CVD_cleaned.csv')
cvd_data = pd.get_dummies(data=cvd_data, columns=['General_Health', 'Checkup', 'Diabetes', 'Age_Category'])
cvd_data.replace({'Yes':1, 'No':0, 'Male':1, 'Female':0}, inplace=True)
cvd_data.rename(columns={'Checkup_5 or more years ago':'Checkup_5+',
                         'Checkup_Within the past 2 years':'Checkup_2',
                         'Checkup_Within the past 5 years':'Checkup_5',
                         'Checkup_Within the past year':'Checkup_1',
                         'Diabetes_No, pre-diabetes or borderline diabetes':'Diabetes_No_border',
                         'Diabetes_Yes, but female told only during pregnancy':'Diabetes_Yes_pregnancy'}, inplace=True)
normal_cols = ['Height_(cm)', 'Weight_(kg)', 'BMI', 'Alcohol_Consumption', 'Fruit_Consumption',
               'Green_Vegetables_Consumption', 'FriedPotato_Consumption']

for column in normal_cols:
  cvd_data[column] = (cvd_data[column] - cvd_data[column].min()) / (cvd_data[column].max() - cvd_data[column].min())

x_columns = cvd_data.columns.tolist()
x_columns.remove('Heart_Disease')

y_columns = ['Heart_Disease']

x_train, x_test, y_train, y_test = train_test_split(cvd_data[x_columns], cvd_data[y_columns], train_size=0.7, random_state=42)

def ml_model(model):
  model.fit(x_train, y_train)
  return model.score(x_test, y_test)

combo_sampler = SMOTETomek()
x_train_combo, y_train_combo = combo_sampler.fit_resample(x_train, y_train)


optimized_params = {
    'criterion': 'entropy',
    'max_depth': 35,
    'max_features': 64,
    'min_samples_leaf': 5,
    'min_samples_split': 9,
    'n_estimators': 78
}

forest_clf = RandomForestClassifier(**optimized_params)

forest_clf.fit(x_train, y_train.to_numpy().ravel())
forest_clf.score(x_test, y_test)
print("Random Forest Classifier Score: " + str(forest_clf.score(x_test, y_test)))
print(cvd_data['Heart_Disease'].sum()) #This will tell me how many Heart_disease yet values there are (because yes = 1, no = 0)
# A common problem with classifiers is predicting all 1s or all 0s (score = number_of_1 / total_number) or (score = 1 - number_of_1 / total_number)

#A Matrix Display for Random Forest

cvd_predictions = forest_clf.predict(x_test)
cvd_cm = confusion_matrix(y_test, cvd_predictions)
tn, fp, fn, tp = cvd_cm.ravel()
print("True N: " + str(tn))
print("True P: " + str(tp))
print("False N: " + str(fn))
print("False P: " + str(fp))

matrix_display = ConfusionMatrixDisplay(cvd_cm)
matrix_display.plot()

plt.savefig('Positive and Negative Heart Disease Diagnosis.png')

plt.show()

#ROC Curve

optimized_xgb_params = {
    'objective': 'binary:logistic',  # Assuming it's a binary classification problem
    'eval_metric': 'logloss',  # You may adjust the metric based on your preference
    'max_depth': 35,
    'colsample_bytree': 0.8,  # You may adjust this based on your preference
    'subsample': 0.8,  # You may adjust this based on your preference
    'learning_rate': 0.1,  # You may adjust this based on your preference
    'min_child_weight': 5,
    'gamma': 0.5,  # You may adjust this based on your preference
    'n_estimators': 78
}

forest_model = XGBClassifier(**optimized_xgb_params)

forest_model.fit(x_train, y_train.to_numpy().ravel())

prediction = forest_model.predict_proba(x_test)[:, 1]
print(prediction)

fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=prediction)

plt.figure()
plt.plot(tpr, tpr, label='No Skill')
plt.plot(fpr, tpr, label='Algorithm')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()

plt.savefig('True (+) vs False (+) Rate in Heart Disease Diagnosis.png')

plt.show()
#%%
"""
Under, Over, and Combo Sampling
"""

#Classify
over_sampler = SMOTE()
x_train_over, y_train_over = over_sampler.fit_resample(x_train, y_train)
print("Oversampler Results: " + str(y_train_over.value_counts()))

under_sampler = RandomUnderSampler()
x_train_under, y_train_under = under_sampler.fit_resample(x_train, y_train)
print("Undersampler Results: " + str(y_train_under.value_counts()))

combo_sampler = SMOTETomek()
x_train_combo, y_train_combo = combo_sampler.fit_resample(x_train, y_train)
print("Combosampler Results: " + str(y_train.value_counts()))

#Graph It

print("Oversampler Gridsearch with Random Forest")
model_clf_over = RandomForestClassifier()
model_clf_over.fit(x_train_over, y_train_over)  # Reshape y_train_over using ravel()
matrix_display = ConfusionMatrixDisplay(cvd_cm)
matrix_display.plot()

plt.savefig('Oversampling HeartDisease.png')

plt.show()

print("Oversampler Gridsearch with XGBClassifier")
model_xgb_over = XGBClassifier()
model_xgb_over.fit(x_train_over, y_train_over)
ConfusionMatrixDisplay.from_estimator(model_xgb_over, x_test, y_test)

plt.show()

print("Undersampler Gridsearch")
model_clf_under = RandomForestClassifier()
model_clf_under.fit(x_train_under, y_train_under) 
ConfusionMatrixDisplay.from_estimator(model_clf_under, x_test, y_test)

plt.savefig('Undersampling HeartDisease.png')

plt.show()

print("Combosampler Gridsearch with Random Forest")
model_clf_combo = RandomForestClassifier()
model_clf_combo.fit(x_train_combo, y_train_combo) 
ConfusionMatrixDisplay.from_estimator(model_clf_combo, x_test, y_test)

plt.show()

print("Combosampler Gridsearch with XGBClassifier")
model_xgb_combo = XGBClassifier()
model_xgb_combo.fit(x_train_combo, y_train_combo)
ConfusionMatrixDisplay.from_estimator(model_xgb_combo, x_test, y_test)


plt.show()
#%%
"""
Thresdhold Tuning
"""

#Increasing the threshold should decrese the number of false positives if your data is good.
#Decresing the threshold should decrese the number of false negatives if you data is good.
import numpy as np
def round_threshold(values, threshold = 0.5):
  """
  This function rounds values to from between 0 and 1 to 1 if the values are
  above the given threshold, or 0 if they are below the threshold.

  Inputs:
  -------
  Values: a numpyt array of floats with shape (nvalues,) to be rounded
  Threshold: float between 0 and 1 which establishes the rounding threshold
  default = 0.5

  Outputs:
  -------
  rounded_values: An array of ints with shape(nvalues,) values rounded based on the threshold
  """

  rounded_values = abs(np.ceil(values - threshold).astype(int))
  return rounded_values

##Random Forest
model_clf_threshold = RandomForestClassifier()
model_clf_threshold.fit(x_train, y_train.to_numpy().ravel())
#predictions = model_clf_threshold.predict(x_test) --> Gives descrete classes, not probabilities
predictions_threshold = model_clf_threshold.predict_proba(x_test)[:, 1] #Rounds the values, so it's not perfect, but it is better.

#predict_proba() returns two: The probability of 0 and the probability of 1

thresholds = np.unique(predictions_threshold)

ConfusionMatrixDisplay.from_estimator(model_clf_under, x_test, y_test)
plt.show()

##XGBoost
model_xgb_threshold = XGBClassifier()
model_xgb_threshold.fit(x_train, y_train.to_numpy().ravel())
#predictions = model_clf_threshold.predict(x_test) --> Gives descrete classes, not probabilities
predictions_threshold = model_xgb_threshold.predict_proba(x_test)[:, 1] #Rounds the values, so it's not perfect, but it is better.

#predict_proba() returns two: The probability of 0 and the probability of 1

thresholds = np.unique(predictions_threshold)

ConfusionMatrixDisplay.from_estimator(model_clf_under, x_test, y_test)
plt.show()


f1_scores = np.zeros((len(thresholds))) #Create an array of zeros to put our scores in

for i in range(len(thresholds)):
  temp_prediction = round_threshold(predictions_threshold, thresholds[i])
  f1_scores[i] = f1_score(y_test, temp_prediction)

print(f1_scores)

#%%
"""
Best Score Index
"""

model_forest_grid_cv = RandomForestClassifier()

# Params of RFC: n_estimators, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, max_leaf_nodes ...

params_dict = {'n_estimators':[10, 55, 100, 200, 350, 500], 'max_depth':[2, 5, 13, 22, 50, 80], 'min_samples_split':[2], 'min_samples_leaf':[1]} #This one takes longer


grid_clf = GridSearchCV(estimator=model_forest_grid_cv, param_grid=params_dict, scoring=None, cv=5)

grid_clf.fit(x_train, y_train)

#show best parameters
best_params = grid_clf.best_params_

#get best scores
best_score = grid_clf.best_score_

print(best_params)
print(best_score)

best_score_index = np.argmax(f1_score) #Returns the index of the biggest value of f1_scores
best_threshold = thresholds[best_score_index]
print(best_threshold)

print("Plot of Threshold Effectivness")
plt.figure()
plt.plot(thresholds, f1_scores)
plt.xlabel('Threshold Value')
plt.ylabel('F1 Score')
plt.title('F1 Score vs Threshold Value')

plt.savefig('Best Threshold Value - Oversampling HeartDisease')

plt.show()
#%%
"""
Apply Threshold Tuning
"""

##RANDOM FOREST

#Oversampler
print("Oversampler Gridsearch with Threshold Tuning - Random Forest")
over_sampler = SMOTE()
x_train_over, y_train_over = over_sampler.fit_resample(x_train, y_train)
print(y_train_over.value_counts())

model_threshold_over = RandomForestClassifier()
model_threshold_over.fit(x_train_over, y_train_over)
predictions_threshold = model_threshold_over.predict_proba(x_test)[:, 1]
threshold_value = 0.2#choosing this because graph below showed highest score around here
binary_predictions = (predictions_threshold >= threshold_value).astype(int)
cvd_cm = confusion_matrix(y_test, binary_predictions)
matrix_display = ConfusionMatrixDisplay(cvd_cm)
matrix_display.plot()
plt.show()

#Undersampler
print("Undersampler Gridsearch with Threshold Tuning - Random Forest")
under_sampler = TomekLinks()
x_train_under, y_train_under = under_sampler.fit_resample(x_train, y_train)
print(y_train_under.value_counts())

model_threshold_under = RandomForestClassifier()
model_threshold_under.fit(x_train_under, y_train_under)
predictions_threshold = model_threshold_under.predict_proba(x_test)[:, 1]
threshold_value = 0.2 #choosing this because graph below showed highest score around here
binary_predictions = (predictions_threshold >= threshold_value).astype(int)

# Calculate the confusion matrix
confusion_matrix_result = confusion_matrix(y_test, binary_predictions)

# Initialize and display the ConfusionMatrixDisplay
matrix_display = ConfusionMatrixDisplay(confusion_matrix_result)
matrix_display.plot()
plt.show()

#Combosampler
print("Combosampler Gridsearch with Threshold Tuning - Random Forest")
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

combo_sampler = SMOTETomek()
x_train_combo, y_train_combo = combo_sampler.fit_resample(x_train, y_train)
print(y_train_combo.value_counts())

model_threshold_combo = RandomForestClassifier()
model_threshold_combo.fit(x_train_combo, y_train_combo)
predictions_threshold = model_threshold_combo.predict_proba(x_test)[:, 1]
threshold_value = 0.2  # choosing this because graph below showed the highest score around here
binary_predictions = (predictions_threshold >= threshold_value).astype(int)

# Calculate the confusion matrix
confusion_matrix_result = confusion_matrix(y_test, binary_predictions)

# Initialize and display the ConfusionMatrixDisplay
matrix_display = ConfusionMatrixDisplay(confusion_matrix_result)
matrix_display.plot()
plt.show()


##XGBoost

#Oversampler
print("Oversampler Gridsearch with Threshold Tuning - XGBoost")
over_sampler = SMOTE()
x_train_over, y_train_over = over_sampler.fit_resample(x_train, y_train)
print(y_train_over.value_counts())

model_threshold_over = XGBClassifier()
model_threshold_over.fit(x_train_over, y_train_over)
predictions_threshold = model_threshold_over.predict_proba(x_test)[:, 1]
threshold_value = 0.2 #choosing this because graph below showed highest score around here
binary_predictions = (predictions_threshold >= threshold_value).astype(int)
cvd_cm = confusion_matrix(y_test, binary_predictions)
matrix_display = ConfusionMatrixDisplay(cvd_cm)
matrix_display.plot()
plt.show()

#Undersampler
print("Undersampler Gridsearch with Threshold Tuning - XGBoost")
under_sampler = TomekLinks()
x_train_under, y_train_under = under_sampler.fit_resample(x_train, y_train)
print(y_train_under.value_counts())

model_threshold_under = XGBClassifier()
model_threshold_under.fit(x_train_under, y_train_under)
predictions_threshold = model_threshold_under.predict_proba(x_test)[:, 1]
threshold_value = 0.2 #choosing this because graph below showed highest score around here
binary_predictions = (predictions_threshold >= threshold_value).astype(int)

# Calculate the confusion matrix
confusion_matrix_result = confusion_matrix(y_test, binary_predictions)

# Initialize and display the ConfusionMatrixDisplay
matrix_display = ConfusionMatrixDisplay(confusion_matrix_result)
matrix_display.plot()
plt.show()

#Combosampler
print("Combosampler Gridsearch with Threshold Tuning - XGBoost")
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

combo_sampler = SMOTETomek()
x_train_combo, y_train_combo = combo_sampler.fit_resample(x_train, y_train)
print(y_train_combo.value_counts())


model_threshold_combo = XGBClassifier()
model_threshold_combo.fit(x_train_combo, y_train_combo)
predictions_threshold = model_threshold_combo.predict_proba(x_test)[:, 1]
threshold_value = 0.2  # choosing this because graph below showed the highest score around here
binary_predictions = (predictions_threshold >= threshold_value).astype(int)

# Calculate the confusion matrix
confusion_matrix_result = confusion_matrix(y_test, binary_predictions)

# Initialize and display the ConfusionMatrixDisplay
matrix_display = ConfusionMatrixDisplay(confusion_matrix_result)
matrix_display.plot()
plt.show()
#%%
"""
XGBoost Active Learning
"""

import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
import matplotlib.pyplot as plt

# Set a random seed for reproducibility
np.random.seed(42)

# Define the base learner (XGBClassifier) and the active learner
optimized_xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 35,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'min_child_weight': 5,
    'gamma': 0.5,
    'n_estimators': 100
}

base_learner = XGBClassifier(**optimized_xgb_params)
learner = ActiveLearner(
    estimator=base_learner,
    query_strategy=uncertainty_sampling,
    X_training=x_train.to_numpy(), y_training=y_train.to_numpy().ravel(),
    verbose=True  # Enable verbose output
)

# Set the number of queries (iterations)
n_queries = 100

# Initialize lists to track performance
query_numbers = []
validation_accuracies = []
uncertainty_scores = []  # Added to track uncertainty scores

# Parameters for adaptive learning rate
initial_learning_rate = 0.1
decay_factor = 0.95

for i in range(n_queries):
    # Update learning rate based on iteration number
    current_learning_rate = initial_learning_rate * (decay_factor ** i)
    learner.estimator.set_params(learning_rate=current_learning_rate)

    # Query random instances
    query_idx = np.random.choice(range(len(x_test)), size=5, replace=False)
    query_instance = x_test.iloc[query_idx].to_numpy()

    # Simulate labeling (replace this with your actual labeling process)
    query_label = y_test.iloc[query_idx].values.reshape(-1,)
    learner.teach(query_instance, query_label)

    # Evaluate the model on the validation set (use x_test for validation in this case)
    validation_accuracy = learner.score(x_test.to_numpy(), y_test.to_numpy().ravel())

    # Track query number, validation accuracy, and uncertainty score
    query_numbers.append(i + 1)
    validation_accuracies.append(validation_accuracy)
    uncertainty_scores.append(np.mean(learner.query_strategy(learner, x_test.to_numpy())))

    print(f'Query {i + 1}/{n_queries} - Validation Accuracy: {validation_accuracy:.4f} - Learning Rate: {current_learning_rate:.4f}')

    # Check for convergence based on early stopping
    if i > 10 and validation_accuracies[i] == validation_accuracies[i-10]:
        print(f'Converged at query {i + 1}')
        break

# Plot learning curve
plt.plot(query_numbers, validation_accuracies, marker='o')
plt.xlabel('Number of Queries')
plt.ylabel('Validation Accuracy')
plt.title('Active Learning Learning Curve')
plt.show()

# Plot uncertainty scores over queries
plt.plot(query_numbers, uncertainty_scores, marker='o')
plt.xlabel('Number of Queries')
plt.ylabel('Uncertainty Score')
plt.title('Model Uncertainty over Queries')
plt.show()

# Final evaluation on the entire test set
final_accuracy = learner.score(x_test.to_numpy(), y_test.to_numpy().ravel())
print(f'Final Accuracy after Active Learning: {final_accuracy:.4f}')


#%%
"""
Ranfom Forest Active Learning
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
import matplotlib.pyplot as plt

# Set a random seed for reproducibility
np.random.seed(42)

# Define the base learner (RandomForestClassifier) and the active learner
base_learner = RandomForestClassifier(n_estimators=50, max_depth=10)
learner = ActiveLearner(
    estimator=base_learner,
    query_strategy=uncertainty_sampling,
    X_training=x_train.to_numpy(), y_training=y_train.to_numpy().ravel()
    # Removed 'verbose' parameter
)

# Set the number of queries (iterations)
n_queries = 50  # Reduced the number of queries for faster execution

# Initialize lists to track performance
query_numbers = []
validation_accuracies = []
uncertainty_scores = []  # Added to track uncertainty scores

for i in range(n_queries):
    # Query the most uncertain instances
    query_idx, query_instance = learner.query(x_test.to_numpy())

    # Simulate labeling (replace this with your actual labeling process)
    query_label = y_test.iloc[query_idx].values.reshape(1, )
    learner.teach(query_instance, query_label)

    # Evaluate the model on the validation set (use x_test for validation in this case)
    validation_accuracy = learner.score(x_test.to_numpy(), y_test.to_numpy().ravel())

    # Track query number, validation accuracy, and uncertainty score
    query_numbers.append(i + 1)
    validation_accuracies.append(validation_accuracy)
    uncertainty_scores.append(np.mean(learner.query_strategy(learner, x_test.to_numpy())))

    print(f'Query {i + 1}/{n_queries} - Validation Accuracy: {validation_accuracy:.4f}')

    # Check for convergence (example: if accuracy doesn't improve after 10 queries)
    if i > 10 and validation_accuracies[i] == validation_accuracies[i-10]:
        print(f'Converged at query {i + 1}')
        break

# Plot learning curve
plt.plot(query_numbers, validation_accuracies, marker='o')
plt.xlabel('Number of Queries')
plt.ylabel('Validation Accuracy')
plt.title('Active Learning Learning Curve')
plt.show()

# Plot uncertainty scores over queries
plt.plot(query_numbers, uncertainty_scores, marker='o')
plt.xlabel('Number of Queries')
plt.ylabel('Uncertainty Score')
plt.title('Model Uncertainty over Queries')
plt.show()

# Final evaluation on the entire test set
final_accuracy = learner.score(x_test.to_numpy(), y_test.to_numpy().ravel())
print(f'Final Accuracy after Active Learning: {final_accuracy:.4f}')
