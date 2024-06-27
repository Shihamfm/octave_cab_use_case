from src.common.utils.datasources import get_cab_usecase_df
from src.common.features.preprocessing.model import *
from pipelines.main import spark,conf,cab_use_case_pd_df

ml_data = cab_use_case_pd_df #Get the pandas dataframe from ML tanks

print("Q4 - Predictive Analysis: \n")

print("Analysis the features\n")

ml_data.info() #Analysing the data types of source data & checking null values

ml_data_with_month = get_month(conf,cab_use_case_pandas_df=ml_data, date_column='Date') #Extracting the month column from date column
ml_data_with_dropColumns =get_dropColumns(conf, cab_use_case_pandas_df=ml_data_with_month, drop_column='Date')

numerical_features = get_numerical_features(conf, cab_use_case_pandas_df=ml_data_with_dropColumns) #Derive the numerical features
categorical_features = get_categorical_features(conf, cab_use_case_pandas_df=ml_data_with_dropColumns) #Derive the categorical features
print(f"Numerical Features: ",numerical_features.columns)
print(f"Categorical Features: ",categorical_features.columns)

#Conver the categorical features into numerical features
ml_data_with_month_one_hot_encoded = get_hot_encoding(conf, cab_use_case_pandas_df=ml_data_with_dropColumns,cat_features=categorical_features.columns)

print("Q4.1 - Split the data: \n")
#split the train and test dataframe
X_train, X_test, y_train, y_test = get_split_train_test(conf, cab_use_case_pandas_df = ml_data_with_month_one_hot_encoded,target_variable='Total_Amount',test_size=0.2,random_state=10)

print("Evaluate the model with linear regression using mean squared error")
#Fit and train the model with linear regression
predict_lr = fit_training_data_linear_regression(X_train=X_train,y_train=y_train,X_test=X_test)

evaluate_model(predicted_data = predict_lr,y_test=y_test)

print("\nEvaluate the model with random forest regressor using mean squared error")
#Fit and train the model with random forest regression
#Let the model to pickup the best max_depth, n_estimators and min_samples_split
predict_rfr = fit_training_data_random_forest(X_train = X_train,
                                                                      y_train = y_train,
                                                                      X_test = X_test,
                                                                      Random_state = 10,
                                                                      max_depth = [5,10,15],
                                                                      n_estimators = [100,250,500],
                                                                      min_samples_split = [3,5,10]
                                                                      )


evaluate_model(predicted_data = predict_rfr,y_test=y_test)

