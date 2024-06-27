#Machine learning libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from src.common.features.preprocessing.main import *

def get_month(conf: dict,cab_use_case_pandas_df: DataFrame,date_column: str) -> DataFrame:
    """
    Extract month column from date
    :param cab_use_case_pandas_df: pandas dataframe
    :param date_column: date column to extract month
    :return: month column
    """
    cab_use_case_pandas_df['Month'] = (
        pd
        .to_datetime(cab_use_case_pandas_df[date_column])
        .dt.strftime('%B')
    )
    return cab_use_case_pandas_df

def get_dropColumns(conf: dict,cab_use_case_pandas_df: DataFrame,drop_column: str) -> DataFrame:
    """
    Drop the columns
    :param cab_use_case_pandas_df: pandas dataframe
    :param drop_column: column to drop
    :return: dataframe after dropping the column
    """
    cab_use_case_pandas_df = (
        cab_use_case_pandas_df
        .drop([drop_column],axis=1)
    )
    return cab_use_case_pandas_df

def get_numerical_features(conf: dict, cab_use_case_pandas_df: DataFrame) -> list:
    """
    Extracting the numerical features
    :param cab_use_case_pandas_df: pandas dataframe
    :return: numerical features
    """
    numerical_features = cab_use_case_pandas_df.select_dtypes(exclude = 'object')

    return numerical_features

def get_categorical_features(conf: dict, cab_use_case_pandas_df: DataFrame) -> list:
    """
    Extracting the categorical features
    :param cab_use_case_pandas_df: pandas dataframe
    :return: numerical features
    """
    categorical_features = cab_use_case_pandas_df.select_dtypes(include = 'object')

    return categorical_features

def get_hot_encoding(conf: dict, cab_use_case_pandas_df: DataFrame,cat_features: list) -> DataFrame:
    """
    Convert the categorical features into numerical
    :param conf: configuration
    :param cab_use_case_pandas_df: pandas dataframe
    :param cat_features: categorical features
    :return: encoding the data source into numerical features
    """

    #initialize one-hot encoder
    encoder = OneHotEncoder(sparse_output=False)

    #Apply the one-hot encoder to the categorical columns
    one_hot_encoded = encoder.fit_transform(cab_use_case_pandas_df[cat_features])

    #Create a datafeame with one-hot encoded columns
    one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(cat_features))

    cab_use_case_pandas_df = pd.concat([cab_use_case_pandas_df, one_hot_encoded_df], axis=1)

    cab_use_case_pandas_df = cab_use_case_pandas_df.drop(cat_features, axis =1)

    return cab_use_case_pandas_df

def get_split_train_test(conf: dict, cab_use_case_pandas_df: DataFrame,target_variable: str,test_size: float,random_state: int) -> DataFrame:
    """
    splitting train and test data
    :param cab_use_case_pandas_df: Pandas dataframe
    :param target_variable: variable to predict
    :param test_size: parameter specifies how large the test set should be
    :param random_state: parameter specifies how large the test set should be
    :return: X_train, X_test, y_train, y_test
    """
    X = cab_use_case_pandas_df.drop([target_variable],axis=1)
    y = cab_use_case_pandas_df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=10)

    return X_train, X_test, y_train, y_test

def fit_training_data_linear_regression(X_train: DataFrame,y_train: DataFrame,X_test: DataFrame) -> float:
    """
    fit the training data set with linear regression model and predict the test data accordingly
    :param X_train: training sample features
    :param y_train: training sample target variable
    :param X_test: testing sample features
    :return: predicted variable according to the linear regression model
    """
    lr = LinearRegression()
    lr.fit(X_train,y_train)
    predict_lr = lr.predict(X_test)

    return predict_lr

def evaluate_model(predicted_data,y_test):
    """

    :param predicted_data: predicted based on the fit model
    :param y_test: True target y variable
    :return: mean_squared_error between predicted and true variable
    """
    mse = mean_squared_error(y_test,predicted_data)
    print(f"MSE: {mse}")

def fit_training_data_random_forest(X_train: DataFrame,y_train: DataFrame,X_test: DataFrame,Random_state: int,max_depth: list,n_estimators: list,min_samples_split: list) -> DataFrame:
    """

    :param X_train: training sample features
    :param y_train: training sample target variable
    :param X_test: testing sample features
    :param Random_state: parameter specifies how large the test set should be
    :param max_depth: maximum tree depth
    :param n_estimators: number of trees
    :param min_samples_split: minimum samples per split
    :return: predicted variable according to the random forest regressor
    """
    RFR = RandomForestRegressor(Random_state)

    param_grid_rfr = {
        'max_depth': max_depth,
        'n_estimators': n_estimators,
        'min_samples_split': min_samples_split
    }
    rfr_cv = GridSearchCV(RFR, param_grid_rfr, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    rfr_cv.fit(X_train, y_train)
    predict_rfr = rfr_cv.predict(X_test)
    print(f"Best estimator of random forest: {rfr_cv.best_estimator_}")

    return predict_rfr