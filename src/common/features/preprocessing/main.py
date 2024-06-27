#Import libraries for analysis

import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from src.common.utils.datasources import get_cab_usecase_df
from src.common.utils.utilities import call_conf
from pyspark.sql import functions as f
from scipy.stats import shapiro, spearmanr


def get_week_weekend_data(conf: dict, cab_use_case_df: DataFrame, date_column: str) -> DataFrame:
    """
    Adding a column for weekdays and weekends
    :param conf: configuration
    :param cab_use_case_df: cab_use_case spark data frame
    :param date_column: date column to extract the day category
    :return: dataframe with weekdays and weekends
    """
    cab_usecase_daycategory = (
        cab_use_case_df
        .withColumn(
            "Days",
            f.dayofweek(date_column)
        )
        .withColumn(
            "Day_Category",
            f.when(
                f.col("Days").isin([1,7]),"Weekend"
            ).otherwise("Weekdays")
        )
        .drop(
            f.col(date_column)
        )
    )
    return cab_usecase_daycategory


def get_night_earnings(conf: dict,cab_use_case_df: DataFrame,time_column: str) -> DataFrame:
    """
    get the night earnings for each Cab_Driver_ID
    :param conf: configuration
    :param cab_use_case_df: cab_use_case spark data frame
    :param time_column: time column to get the Night or Day
    :return: night_earnings for each Cab_Driver_ID
    """
    night_earnings = (
        cab_use_case_df
        .filter(f.col(time_column)=="Night")
        .groupBy(
            "Cab_Driver_ID"
        )
        .agg(
        f.avg(
            f.col("Total_Amount")).alias("Total_Amount")
        )
    )
    return night_earnings


def get_gender_distr(conf: dict,cab_use_case_df: DataFrame,gender_column: str) -> DataFrame:
    """
    get gender distribution for total fair
    :param conf: configuration
    :param cab_use_case_df: cab_use_case spark data frame
    :return: gender_distr
    """
    gender_distr = (
        cab_use_case_df
        .groupBy(
            f.col(gender_column)
        )
        .agg(
            f.avg(
                f.col("Total_Amount")
            ).alias("Total_Amount")
        )
    )
    return gender_distr


def get_week_weekend_earnings(conf: dict,week_weekend_df: DataFrame) -> DataFrame:
    """
    get weekday & weekends category_earnings
    :param conf: configuration
    :param week_weekend_df: week_weekend spark data frame
    :return: Weekday & Weekends category_earnings
    """
    week_weekend_earnings = (
        week_weekend_df
        .groupBy(
            f.col("Day_Category")
        )
        .agg(
            f.avg(
                f.col("Total_Amount")
            ).alias('Avg Total Amount')
        )
    )
    return week_weekend_earnings


def get_profitable_state(conf: dict,cab_use_case_df):
    """
    get profitable_states by PickUp_Colombo_ID
    :param conf: configuration
    :param cab_use_case_df: cab_use_case spark data frame
    :return: profitable_states by PickUp_Colombo_ID
    """
    profitable_state = (
        cab_use_case_df
        .groupBy(
            f.col("PickUp_Colombo_ID")
        )
        .agg(
            f.sum(
                f.col("Total_Amount")
            ).alias("Total_Amount")
        )
        .orderBy(
            f.desc(
                f.col("Total_Amount")
            )
        )
    )
    return profitable_state

def get_profitable_distr(conf: dict,cab_use_case_df: DataFrame ,gender: str,time: str) -> DataFrame:
    """
    get the best state by gender and time
    :param gender: Gender
    :param time: Pickup time
    :param conf: configuration
    :param cab_use_case_df: cab_use_case spark data frame
    :return: profitable_dist
    """
    profitable_dist = (
        cab_use_case_df
        .filter(
            f.col("Gender")==gender)
        .filter(
            f.col("PickUp_Time")==time)
        .groupBy(
            f.col("Gender"),
            f.col("PickUp_Time"),
            f.col("PickUp_Colombo_ID")
        )
        .agg(
            f.sum(
                f.col("Total_Amount")
            ).alias("Total_Amount"))
        .orderBy(
            f.desc(
                f.col("Total_Amount")
            )
        )
        .limit(1)
    )
    return profitable_dist

def get_pivot_tiprates(conf: dict, cab_use_case_pandas_df: DataFrame, x_axis: str, y_axis: str, values: str, agg: str) -> DataFrame:
    """
    Derive the pivot table of the average tip rates
    :param x_axis: x_axis of the table
    :param y_axis: y_axis of the table
    :param values: tip rates
    :param conf: configuration
    :param cab_use_case_pandas_df: cab_use_case pandas data frame
    :param agg: aggregation of the values
    :return: pivot table of average tip rates
    """
    pivot_tip_rates = (
        cab_use_case_pandas_df
        .pivot_table(
            index=y_axis,
            columns=x_axis,
            values=values,
            aggfunc=agg
        ).round(2)
    )
    return pivot_tip_rates


def get_hypothesis_testing_normal_distr(conf: dict, cab_use_case_pandas_df: DataFrame, target: str) -> DataFrame:
    """
    Doing a Hypothesis test for if the target variable is normally distributed or not
    :param cab_use_case_pandas_df: cab_use_case pandas data frame
    :param target: target numerical variable
    :return: statement normal distribution or not
    """
    stat, p = shapiro(cab_use_case_pandas_df[target])

    print(f'Null Hypothesis: {target} is normally distributed')
    print(f'Alternative Hypothesis: {target} is not normally distributed\n')
    print('stat: %.3f, p: %.30f' % (stat, p))
    if p > 0.05:
        print('Normal distribution\n')
    else:
        print('Not a normal distribution\n')

def get_hypothesis_testing_correlation(cab_use_case_pandas_df: DataFrame, target_1: str, target_2: str) -> DataFrame:
    """
    Doing a hypothesis testing for an argument of correlation
    :param cab_use_case_pandas_df: cab_use_case pandas data frame
    :param target_1: target numerical variable_1
    :param target_2: target numerical variable_2
    :return: correlation between two variables
    """
    stat, p = spearmanr(cab_use_case_pandas_df[target_1], cab_use_case_pandas_df[target_2])

    print(f"Null Hypothesis: There's a correlation between {target_1} & {target_2}")
    print(f"Alternative Hypothesis: There's no correlation between {target_1} & {target_2}\n")
    print('stat: %.3f, p: %.30f' % (stat, p))
    if p > 0.05:
        print('Independent sample - can reject the null hypothesis\n')
    else:
        print("Dependent sample - can't reject the null hypothesis\n")
