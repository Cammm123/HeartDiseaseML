#%%
"""
SET UP
"""
#Still need to run import cell
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
#%%
#CELL 1
"""
Hyperparameter Tuning--------------------------------------------------------------------
"""
#%%
#CELL 3
# Import necessary libraries
from skopt import BayesSearchCV
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense

# Define hyperparameter search spaces
rf_params = {
    'n_estimators': (10, 100),
    'max_features': (1, 64),
    'max_depth': (5, 50),
    'min_samples_split': (2, 11),
    'min_samples_leaf': (1, 11),
    'criterion': ['gini', 'entropy']
}

svm_params = {
    'C': (0, 50),
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}

ann_params = {
    'optimizer': ['adam', 'rmsprop', 'sgd'],
    'activation': ['relu', 'tanh'],
    'batch_size': [16, 32, 64],
    'neurons': sp_randint(10, 100),
    'epochs': [20, 50],
    'patience': sp_randint(3, 20)
}

# Random Forest hyperparameter tuning
clf_rf = RandomForestClassifier(random_state=0)
hyper_rf = BayesSearchCV(clf_rf, search_spaces=rf_params, cv=3, n_iter=10, scoring='accuracy')
hyper_rf.fit(x_train_combo, y_train_combo.squeeze())  # Reshape y_train_combo using squeeze()
print(hyper_rf.best_params_)
print("Accuracy: " + str(hyper_rf.best_score_))  

# SVM hyperparameter tuning
clf_svm = SVC(gamma='scale')
hyper_svm = BayesSearchCV(clf_svm, search_spaces=svm_params, cv=3, n_iter=50, scoring='accuracy')
hyper_svm.fit(x_train_combo, y_train_combo.squeeze())  # Reshape y_train_combo using squeeze()
print(hyper_svm.best_params_)
print("Accuracy: " + str(hyper_svm.best_score_))

# ANN hyperparameter tuning
def create_ann(optimizer, activation, batch_size, neurons, epochs, patience):
    model = Sequential()
    model.add(Dense(neurons, input_dim=x_train_combo.shape[1], activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

clf_ann = KerasClassifier(build_fn=create_ann, epochs=20, verbose=0)
hyper_ann = BayesSearchCV(clf_ann, search_spaces=ann_params, cv=3, n_iter=10, scoring='accuracy')
hyper_ann.fit(x_train_combo, y_train_combo.squeeze())  # Reshape y_train_combo using squeeze()
print(hyper_ann.best_params_)
print("Accuracy: " + str(hyper_ann.best_score_))
#%%
import matplotlib.pyplot as plt
from skopt import BayesSearchCV
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from skopt.plots import plot_objective

# Define hyperparameter search spaces
rf_params = {
    'n_estimators': (10, 100),
    'max_features': (1, 64),
    'max_depth': (5, 50),
    'min_samples_split': (2, 11),
    'min_samples_leaf': (1, 11),
    'criterion': ['gini', 'entropy']
}

svm_params = {
    'C': (0, 50),
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}

ann_params = {
    'optimizer': ['adam', 'rmsprop', 'sgd'],
    'activation': ['relu', 'tanh'],
    'batch_size': [16, 32, 64],
    'neurons': sp_randint(10, 100),
    'epochs': [20, 50],
    'patience': sp_randint(3, 20)
}

# Random Forest hyperparameter tuning
clf_rf = RandomForestClassifier(random_state=0)
hyper_rf = BayesSearchCV(clf_rf, search_spaces=rf_params, cv=3, n_iter=5, n_jobs=-1, scoring='accuracy')
hyper_rf.fit(x_train_combo, y_train_combo.squeeze())  # Reshape y_train_combo using squeeze()
print("Random Forest - Best Params:", hyper_rf.best_params_)
print("Accuracy: " + str(hyper_rf.best_score_))

# SVM hyperparameter tuning
clf_svm = SVC(gamma='scale')
hyper_svm = BayesSearchCV(clf_svm, search_spaces=svm_params, cv=3, n_iter=5, n_jobs=-1, scoring='accuracy')
hyper_svm.fit(x_train_combo, y_train_combo.squeeze())  # Reshape y_train_combo using squeeze()
print("SVM - Best Params:", hyper_svm.best_params_)
print("Accuracy: " + str(hyper_svm.best_score_))

# ANN hyperparameter tuning
def create_ann(optimizer, activation, batch_size, neurons, epochs, patience):
    model = Sequential()
    model.add(Dense(neurons, input_dim=x_train_combo.shape[1], activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

clf_ann = KerasClassifier(build_fn=create_ann, epochs=20, verbose=0)
hyper_ann = BayesSearchCV(clf_ann, search_spaces=ann_params, cv=3, n_iter=5, n_jobs=-1, scoring='accuracy')
hyper_ann.fit(x_train_combo, y_train_combo.squeeze())  # Reshape y_train_combo using squeeze()
print("ANN - Best Params:", hyper_ann.best_params_)
print("Accuracy: " + str(hyper_ann.best_score_))

# Visualize results
results = [hyper_rf, hyper_svm, hyper_ann]
model_names = ['Random Forest', 'SVM', 'ANN']

for model, name in zip(results, model_names):
    plot_objective(model, title=name + ' Hyperparameter Tuning', ylabel='Accuracy')

plt.show()
