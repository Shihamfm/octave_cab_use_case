import os
import yaml
from typing import Dict, Mapping
from pyspark.sql import functions as f


def call_conf(conf_path: str = "../../../../conf/conf.yml") -> Dict:
    """
    :return: configuration file
    """
    with open(conf_path) as file:
        #The fullloader paramenter handles the conversion from YAML
        #Scaler values to Python dictionary format
        conf = yaml.load(file,Loader=yaml.FullLoader)

    return conf