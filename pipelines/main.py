#import os
from src.common.features.preprocessing.main import *
from src.common.utils.datasources import get_cab_usecase_df
from src.common.utils.utilities import call_conf
from pyspark.sql import functions as f
from pyspark.sql import SparkSession

spark = SparkSession.builder\
    .appName('cab_usecase')\
   .getOrCreate()

conf = call_conf("../conf/conf.yml")

cab_use_case_df = get_cab_usecase_df(conf,spark) #load the spark dataframe
cab_use_case_pd_df = cab_use_case_df.toPandas() #conver the spark dataframe into pandas dataframe

cab_usecase_daycategory = get_week_weekend_data(conf, cab_use_case_df, date_column='Date')
night_earnings = get_night_earnings(conf, cab_use_case_df,time_column='PickUp_Time')
gender_distribution = get_gender_distr(conf, cab_use_case_df, gender_column = 'Gender')
daycategory_distribution = get_week_weekend_earnings(conf, week_weekend_df = cab_usecase_daycategory)
profitable_state = get_profitable_state(conf, cab_use_case_df)

print("Q3.A - The drivers make each night on average: ")
night_earnings.show()

print("Q3.A - Total Amount distribution for Male and Female: ")
gender_distribution .show()

print("Q3.B - Average amount earned between weekdays and weekends: ")
daycategory_distribution.show()

print("Q3.C - The best Colombo states in Colombo for drivers to be in order to pick up profitable fares: ")
profitable_state.show()

print("The best Colombo state in Colombo for drivers to be in order to pick up profitable fares:")
print("For Gender: M & time: Day")
gender = 'M'
time = 'Day'
get_profitable_distr(conf, cab_use_case_df, gender,time,).show()

print("For Gender: F & time: Day")
gender = 'F'
time = 'Day'
get_profitable_distr(conf, cab_use_case_df, gender,time,).show()

print("For Gender: M & time: Night")
gender = 'M'
time = 'Night'
get_profitable_distr(conf, cab_use_case_df, gender,time).show()

print("For Gender: F & time: Night")
gender = 'F'
time = 'Night'
get_profitable_distr(conf, cab_use_case_df, gender,time).show()

print("Q3.D - Tipping rates vary by Colombo state ID")

print('Average Tipping rates vary by Colombo state ID')
x_axis = 'DropOff_Colombo_ID'
y_axis = 'PickUp_Colombo_ID'
values = 'Tip'
agg = 'mean'
print(get_pivot_tiprates(conf, cab_use_case_pd_df,x_axis, y_axis, values, agg))


print('Count of Tipping rates received vary by Colombo state ID')
x_axis = 'DropOff_Colombo_ID'
y_axis = 'PickUp_Colombo_ID'
values = 'Tip'
agg = 'count'
print(get_pivot_tiprates(conf, cab_use_case_pd_df,x_axis, y_axis, values, agg))


print("Q3.E - Explorative data analysis\n")

print("__Total fair distribution__\n")
target  = 'Total_Amount'
get_hypothesis_testing_normal_distr(conf, cab_use_case_pd_df,target)

print("__Correlation between Total fair and tip__\n")
target_1  = 'Total_Amount'
target_2  = 'Tip'
get_hypothesis_testing_correlation(cab_use_case_pd_df,target_1,target_2)
