import pandas as pd
import numpy as np

train_data = pd.read_csv('/Users/aishwaryapirankar/Desktop/Bootcamp Projects/HR - PS 4/Data/train.csv')
test_data = pd.read_csv('/Users/aishwaryapirankar/Desktop/Bootcamp Projects/HR - PS 4/Data/test.csv')

# Get total missing values in 'education' column
train_data.education.isnull().sum()

def fill_missing(df, column, value):
    df_copy = df.copy()
    df_copy[column].fillna(value=value,inplace=True)
    return df_copy

train_data = fill_missing(train_data, "education", "others")
test_data = fill_missing(test_data, "education", "others")

train_data = fill_missing(train_data, "previous_year_rating", 0.0)
test_data = fill_missing(test_data, "previous_year_rating", 0.0)

# Convert 'object' columns into categorical datatype

def convert_to_category(df, column_list):
    df_copy = df.copy()
    df_copy[column_list] = df_copy[column_list].apply(lambda x: x.astype('category'))
    return df_copy

train_data = convert_to_category(train_data, ['department','region','education','gender','recruitment_channel'])
test_data = convert_to_category(test_data, ['department','region','education','gender','recruitment_channel'])

# Check if there is any duplicate employee ID
# Could also use df.duplicated().sum()

condition = bool(train_data.duplicated(subset = 'employee_id').any())

if condition:
    print('There are duplicate employee IDs')
else:
    print('No duplicate employee IDs')

#EDA

# Check class balance

train_data.is_promoted.value_counts(normalize=True)

# Model Preparation

# Feature Engineering
# Bin ‘AGE’ data to groups
# Transform both training data and test data

def convert_age_to_group(df):
    df_copy = df.copy()
    bins = range(20,61,5)    # every 5 years as a bin
    labels = list(range(len(bins)-1))
    df_copy['age_group'] = pd.cut(df_copy['age'],bins=bins, labels=labels, right=True, include_lowest=True)
    df_copy.drop(columns=["age"], inplace=True)
    return df_copy

train_data = convert_age_to_group(train_data)
test_data = convert_age_to_group(test_data)

# Use LabelEncoder to convert categorical features into numerical array

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

def convert_to_numerical(df_train, df_test):
    df_train_copy = df_train.copy()
    df_test_copy = df_test.copy()
    
    for i in ["department", "region", "education", "gender", "recruitment_channel", "age_group"]:
        le = LabelEncoder()
        le.fit(df_train_copy.loc[:, i])
        df_train_copy.loc[:, i] = le.transform(df_train_copy.loc[:, i])
        df_test_copy.loc[:, i] = le.transform(df_test_copy.loc[:, i])
    
    return df_train_copy, df_test_copy

train_data, test_data = convert_to_numerical(train_data, test_data)

# Feature Selection

# Define predictor variables and target variable
X = train_data.drop(columns=['is_promoted'])
y = train_data['is_promoted']

X_test = test_data.copy()

# Save all feature names as list
feature_cols = X.columns.tolist() 

# Extract numerical columns and save as a list for rescaling
num_cols = ['no_of_trainings', 'previous_year_rating', 'length_of_service', 'awards_won?', 'avg_training_score']

# Split training and test data

# Define function to split data with and without SMOTE 

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def data_split(X, y, imbalance = False):
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.3,shuffle=True, stratify=y, random_state=42)
    
    if imbalance:
        # use SMOTE to over sample data
        sm = SMOTE(random_state = 42)
        X_train, y_train = sm.fit_resample(X_train, y_train.ravel())
    
    return X_train, X_validation, y_train, y_validation

# Rescale Features

# Define function to rescale training data using StandardScaler

from sklearn.preprocessing import StandardScaler

def standard_scaler(X_train, X_validation, X_test,  numerical_cols):
    
    # Make copies of dataset
    X_train_std = X_train.copy()
    X_validation_std = X_validation.copy()
    X_test_std = X_test.copy()
    
    # Apply standardization on numerical features only
    for i in numerical_cols:
        scl = StandardScaler().fit(X_train_std[[i]])     # fit on training data columns
        X_train_std[i] = scl.transform(X_train_std[[i]]) # transform the training data columns
        X_validation_std[i] = scl.transform(X_validation_std[[i]])   # transform the validation data columns
        X_test_std[i] = scl.transform(X_test_std[[i]])   # transform the test data columns

    return X_train_std, X_validation_std, X_test_std

# Model Building

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import joblib

def run_models(X, y, X_test, num_cols, models):
    
    model_result = []

    for imbalance in [True, False]:
        X_train, X_validation, y_train, y_validation = data_split(X, y, imbalance = imbalance)
        X_train_std, X_validation_std, X_test_std = standard_scaler(X_train, X_validation, X_test, numerical_cols = num_cols)
       
        # Fit the model
        for model_name, model in models.items():
            model.fit(X_train_std, y_train)
            #joblib.dump(model, f"{model_name}.pkl")   # save models as pickle file
            scores = cross_val_score(model, X_train_std, y_train, scoring ="roc_auc", cv = 5)
            roc_auc = np.mean(scores)

            model_result.append([model_name, imbalance,  roc_auc]) 
    df = pd.DataFrame(model_result, columns = ["Model", "SMOTE" , "ROC_AUC Score"])  
    df.to_csv("model_initial.csv", index=None)
    
    return df

# # Fit multipe models with and without SMOTE sampling

# model_dict = {"Logistic Regression":LogisticRegression(random_state=42), 
#               "Random Forest":RandomForestClassifier(random_state=42), 
#               "XGBoost":  XGBClassifier(random_state=42)}

# run_models(X, y, X_test, num_cols, model_dict)

# Hyperparameter Tuning

# Logistic Regression

# Randomized search for the best C parameter

# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import uniform

# # Split data with SMOTE 
# X_train, X_validation, y_train, y_validation = data_split(X, y, imbalance = True)

# # Rescale data
# X_train_std, X_validation_std, X_test_std = standard_scaler(X_train, X_validation, X_test, numerical_cols = num_cols)

# # Fit the model
# logistic = LogisticRegression(solver='saga', tol=1e-2, max_iter=200,random_state=42)
# distributions = dict(C=uniform(loc=0, scale=4), penalty=['l2', 'l1'])
# lr_best = RandomizedSearchCV(logistic, distributions, random_state=42)

# lr_best= lr_best.fit(X_train_std, y_train)   

# print(lr_best.best_params_)

# Save tuned model and parameters

#joblib.dump(lr_best,"logreg_tuned.pkl")

# # Get ROC_AUC score of tuned model on validation data

# scores_tuned = cross_val_score(lr_best, X_validation_std, y_validation, scoring = "roc_auc", cv = 5)
# roc_auc_lr_best = np.mean(scores_tuned)

# # Save best ROC_AUC 
# #joblib.dump(roc_auc_lr_best,"logreg_ROC_AUC_tuned.pkl")  

# print(f'ROC_AUC score after tuning parameters:{roc_auc_lr_best:.3f}')

# Random Forest

# from sklearn.model_selection import GridSearchCV

# # Split data with SMOTE 
# X_train, X_validation, y_train, y_validation = data_split(X, y, imbalance = True)

# # Create parameter grid  
# param_grid = {
#     'max_depth': [60, 90, 110],
#     'min_samples_leaf': [3, 4, 5],
#     'min_samples_split': [8, 10, 12],
#     'n_estimators': [100, 200, 300]
# }

# # Instantiate the model
# clf_rf = RandomForestClassifier(random_state=42)

# # Instantiate grid search model
# rf_best = GridSearchCV(estimator = clf_rf, param_grid = param_grid,    
#                           cv = 3, n_jobs = -1, verbose = 1)

# # Fit grid search to the data
# rf_best.fit(X_train, y_train)
# rf_best.best_params_

# Save tuned model and parameters

#joblib.dump(rf_best,"clf_rf_tuned.pkl")

"""Tune the model one more time since the best parameters are the minimum or the maximum values provided."""

# from sklearn.model_selection import GridSearchCV

# # Split data with SMOTE 
# X_train, X_validation, y_train, y_validation = data_split(X, y, imbalance = True)

# # Create parameter grid  
# param_grid = {
#     'max_depth': [50,60,70],
#     'min_samples_leaf': [2,3],
#     'min_samples_split': [6,7,8],
#     'n_estimators': [200,300,400]
# }

# # Instantiate the model
# clf_rf = RandomForestClassifier(random_state=42)

# # Instantiate grid search model
# rf_best1 = GridSearchCV(estimator = clf_rf, param_grid = param_grid,    
#                           cv = 3, n_jobs = -1, verbose = 1)

# # Fit grid search to the data
# rf_best1.fit(X_train, y_train)
# rf_best1.best_params_

# # Get ROC_AUC score of tuned model on validation data

# scores_tuned = cross_val_score(rf_best1, X_validation, y_validation, scoring = "roc_auc", cv = 5)
# roc_auc_rf_best = np.mean(scores_tuned)

# # Save best ROC_AUC 
# #joblib.dump(roc_auc_rf_best,"rf_ROC_AUC_tuned.pkl") 

# print(f'ROC_AUC score after tuning parameters:{roc_auc_rf_best:.3f}')

# XG Boost

from pprint import pprint

# Number of trees
n_estimators = np.arange(200,1000,200)

# Minimum loss reduction required to make a further partition on a leaf node of the tree
# The larger gamma is, the more conservative the algorithm will be
gamma = np.arange(0.1,0.6,0.1)

# Default 0.3, range(0,1)
learning_rate = np.arange(0.1,0.6,0.1)

# Maximum number of levels in tree
max_depth = list(range(3,8,1))

# Subsample ratio of the training instances.Range(0,1)
subsample = np.arange(0.5,0.9,0.1)

# Subsample ratio of columns when constructing each tree. Range(0,1)
colsample_bytree = np.arange(0.5,0.9,0.1)

# Control the balance of positive and negative weights
# Sum(negative instances) / sum(positive instances)
scale_pos_weight = [1,3.5]


# Create the random grid
random_grid_xgb = {'n_estimators': n_estimators,
                   'gamma': gamma,
                   'learning_rate':learning_rate,
                   'max_depth': max_depth,
                   'subsample':subsample,
                   'colsample_bytree':colsample_bytree,
                   'scale_pos_weight':scale_pos_weight
                  }
pprint(random_grid_xgb)

# Split data with SMOTE 
from sklearn.model_selection import RandomizedSearchCV

X_train, X_validation, y_train, y_validation = data_split(X, y, imbalance = True)
xgboost = XGBClassifier()

# Use randomized search
xgb_random = RandomizedSearchCV(estimator = xgboost, 
                                param_distributions = random_grid_xgb, 
                                n_iter = 10, 
                                cv = 3, 
                                verbose=1, 
                                random_state=42, 
                                n_jobs = -1,
                                scoring ='roc_auc')

xgb_random.fit(X_train, y_train)    

# print(xgb_random.best_params_,xgb_random.best_score_)

# Save tuned model and parameters 
import pickle 
pickle.dump(xgb_random, open("xgb_tuned.pkl","wb"))

# Get ROC_AUC score of tuned model on validation data

# scores_tuned = cross_val_score(xgboost, X_validation, y_validation, scoring = "roc_auc", cv = 5)
# roc_auc_xgb_best = np.mean(scores_tuned)

# Save best ROC_AUC 
#joblib.dump(xgb_random,"xgb_ROC_AUC_tuned.pkl") 

# print(f'ROC_AUC score after tuning parameters:{roc_auc_xgb_best:.3f}') 

# Model Performance Evaluation

# Define a function to compute Precision, Recall and F1 score

from sklearn.metrics import confusion_matrix

def get_pre_rec_f1(model_name, model,X_validation,y_validation):
    y_pred = model.predict(X_validation)
    tn, fp, fn, tp = confusion_matrix(y_validation, y_pred).ravel()
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    F1 = 2 * (precision * recall) / (precision + recall)
    accuracy = (tp + tn) / (tn + fp + tp + fn)
    
    return [model_name, precision, recall, F1, accuracy]

# # Logistic Regression model performance on validation data:
# col_1 = get_pre_rec_f1("Logistic", lr_best, X_validation_std, y_validation)

# # Random Forest model performance on validation data:
# col_2 = get_pre_rec_f1("Random Forest",  rf_best1, X_validation, y_validation)

# XGBoost model performance on validation data:
# col_3 = get_pre_rec_f1("XGBoost", xgb_random, X_validation, y_validation)

# result = []
# result.append(col_1)
# result.append(col_2)
# result.append(col_3)

# pd.DataFrame(result, columns = ["Model", "Precision", "Recall", "F1", "Accuracy"])

#%pip install scikit-plot
# Plot ROC_AUC curve of 3 models
# from sklearn import plot_roc_curve

# fig,ax=plt.subplots(figsize=(10,5))

# plot_roc_curve(lr_best, X_validation_std, y_validation,ax=ax, color="blue",label='Logistic Regression')
# plot_roc_curve(rf_best1, X_validation, y_validation,ax=ax, color="black",label='Random Forest')
# plot_roc_curve(xgb_random, X_validation, y_validation,ax=ax, color="red",label='XGBoost')

# plt.title('ROC/AUC of 3 models')
# plt.grid()

# The model with best F1 is XGBoost.

# Model Prediction


# Make prediction using the best model XGBoost
y_prediction = xgb_random.predict(X_test) 
print(X_test.head()) 
result_submission = pd.DataFrame({"employee_id": test_data.employee_id, "is_promoted": y_prediction})
result_submission.to_csv("submission1.csv", index=None)

