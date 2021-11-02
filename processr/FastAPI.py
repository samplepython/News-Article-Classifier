import os
import uvicorn
import requests
import json
from bson import json_util
from bson.json_util import dumps, RELAXED_JSON_OPTIONS
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from ProcessorUtility import process_train_data, process_predict_data_str
from pyspark.sql.functions import to_json,col
from TrainModelUtility import train_model
from PredictModelUtility import load_model, ProcessUrl, predict
from pyspark.sql import SparkSession,SQLContext
import pyspark.sql.types as tp 
import pandas as pd

TRAINR_ENDPOINT = os.getenv("TRAINR_ENDPOINT")

# defining the main app
app = FastAPI(title="processor", docs_url="/")

# class which is expected in the payload while training
class QueryOut(BaseModel):
   topic: str

class QueryIn(BaseModel):
   url: str

class PageSummary(BaseModel):
   summary: str
   def __init__(self, mystr=str()):
         self.summary = mystr
         
def read_from_csv(csv_file_path):
    my_spark = SparkSession\
	     .builder\
	     .appName("Python Spark train News Classifier")\
	     .getOrCreate() 
    # define the schema
    my_schema = tp.StructType([
        tp.StructField(name= 'topic',        dataType= tp.StringType(),    nullable= False),
        tp.StructField(name= 'title', 	     dataType= tp.StringType(),    nullable= True),
        tp.StructField(name= 'summary',      dataType= tp.StringType(),   nullable= False)
        ])
    # read the data with the defined schema
    my_data = my_spark.read.format('com.databricks.spark.csv')\
                .options(header='true', schema=my_schema)\
                .load(csv_file_path,header=True);
    # print the schema
    my_data.printSchema()
    # get the dimensions of the data
    #print("Rows:"+str(my_data.count())+" Cols:"+str(len(my_data.columns)))
    return my_data

# Route definitions
@app.get("/ping")
# Healthcheck route to ensure that the API is up and running
def ping():
    return {"ping": "pong"}


@app.post("/process", status_code=200)
# Route to take in data from MongoDB, process it and send it for training.
def process():
    df=read_from_csv("Data/train.csv")    
    processed_df = process_train_data(df)
    # send the processed data to trainr for training  
    train_model(processed_df)   
    return {"detail": "Processing successful"}

def read_from_predict_csv(csv_file_path):
    my_spark = SparkSession\
	     .builder\
	     .appName("Python Spark train News Classifier")\
	     .getOrCreate() 
    # define the schema
    my_schema = tp.StructType([
        tp.StructField(name= 'summary', dataType= tp.StringType(), nullable= False)
        ])
    # read the data with the defined schema
    my_data = my_spark.read.format('com.databricks.spark.csv')\
                .options(header='true', schema=my_schema)\
                .load(csv_file_path,header=True);
    # print the schema
    my_data.printSchema()
    # get the dimensions of the data
    #print("Rows:"+str(my_data.count())+" Cols:"+str(len(my_data.columns)))
    return my_data


@app.post("/predict_news", response_model=QueryOut, status_code=200)
# Route to do the prediction using the ML model defined.
# Payload: QueryIn containing the parameters
# Response: QueryOut containing the flower_class predicted (200)
def predict_news(query_data: QueryIn):   
    summary = ProcessUrl(query_data) 
    processed_df = process_predict_data_str(summary)    
    processed_df.show()
    output = {"topic": predict(processed_df)}    
    return output

# Main function to start the app when main.py is called
if __name__ == "__main__":
    # Uvicorn is used to run the server and listen for incoming API requests on 0.0.0.0:8888
    uvicorn.run("main:app", host="0.0.0.0", port=8888, reload=True)
