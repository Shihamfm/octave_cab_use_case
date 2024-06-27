import os
from src.common.utils.utilities import call_conf
from pyspark.sql import SparkSession, DataFrame


def get_cab_usecase_df(conf: dict, spark: SparkSession) -> DataFrame:
    """
    This is the function for loading cab use case data from local location

    These are the columns of csv file
    Cab_Driver_ID:	    A unique cab driver id
    Gender:     	    Gender of the driver
    PickUp_Colombo_ID:	Colombo State ID (Eg: Colombo -1, Colombo - 2 etc )
    N_Passengers:   	Number of passengers got into the cab
    Date:       	    Pick up date
    PickUp_Time:    	Pick up time. Indicates whether its a day time or night time
    Duration_Min:   	Total duration in min
    Tip:            	Tip that passenger paid in addition to the fair in $
    Total_Amount:   	Total amount that driver received in $

    :param spark: spark
    :param conf: configuration
    :return: convert the csv file to spark dataframe
    """
    cab_use_case_data_df = (
        spark.read
        .csv(
            os.path.join(conf['paths']['file_path'],'Colombo-Cab-data-2.csv'),
            header=True,
            inferSchema=True
        )
    )
    return cab_use_case_data_df