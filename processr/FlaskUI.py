from datetime import datetime
from flask import Flask, render_template,request,redirect,url_for # For flask implementation
from pymongo import MongoClient
from ProcessorUtility import process_predict_data_str,process_train_data
from pydantic import BaseModel
from PredictModelUtility import ProcessUrl, predict,retrain_model
from pyspark.sql.types import *
from pyspark.sql.functions import * 
from pyspark.sql import SparkSession
import pyspark.sql.types as tp 
from TrainModelUtility import train_model
import os

app = Flask(__name__)
heading = "News Article Classifier"
title = "News Article classification"

mongo_url = os.getenv('ME_CONFIG_MONGODB_URL', 'mongodb://root:example@mongo:27017/')
 
client = MongoClient(mongo_url) #host uri
db = client.newsclassifier    #Select the database
todos = db.predictionresults #Select the collection name
retraincol = db.retraincollection
todos_new={} #for current value

# URL provided for prediction
class QueryIn(BaseModel):
   url: str 

#redirect url
def redirect_url():
    return request.args.get('next') or \
           request.referrer or \
           url_for('index')

#list the prediction history
@app.route("/list")
def lists ():
	#Display the all Tasks
	todos_l = todos.find().sort("date",-1)
	todosnew_l={"Category":""}
	a1="active"
	return render_template('index.html',a1=a1,todos=todos_l,todos_new=todosnew_l,t=title,h=heading)

#handle invalid requests
@app.route("/invalid")
def invalidurl ():
	#Display the all Tasks	
	todos_l = todos.find().sort("date",-1)
	todosnew_l={"Category":"Invalid News Article URL"}
	a1="active"	
	return render_template('index.html',a1=a1,todos=todos_l,todos_new=todosnew_l,t=title,h=heading)

#home page
@app.route("/")
@app.route("/uncompleted")
def tasks ():
	#Display the Uncompleted Tasks
	todos_l = ["url"]
	todosnew_l={"Category":""}
	a2="active"
	return redirect("/list") #render_template('index.html',a2=a2,todos=todos_l,todos_new=todosnew_l,t=title,h=heading)

# read the train data csv file
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
                .load(csv_file_path,header=True)
    # print the schema
    my_data.printSchema()
    # get the dimensions of the data
    #print("Rows:"+str(my_data.count())+" Cols:"+str(len(my_data.columns)))
    return my_data

# Verify if URL is already proceed by the classifier or not
def validateurl(name):
	ret_val=False	
	if todos.count_documents({ "_id":name }):
		ret_val=True
	return ret_val        

#Handle the Classify button action
@app.route("/action", methods=['POST'])
def action ():
	#make prediction
	a1="active"
	todosnew_l={}
	option = request.form['options']
	name=str(request.values.get("url"))
	print('selected option is: '+ option)
	if option == 'Single':
		predval=''		
		if validateurl(name)==False:
			query_data= QueryIn(url=name)
			try:
				summary = ProcessUrl(query_data)
				if len(summary) == 0:
					return redirect("/invalid")
					#todosnew_l={"Category":name+": invalid News Article URL"}
				else:
					df=process_predict_data_str(summary)
					df.show()   
					#insert into mongodb
					predval=predict(df)			
					todos.insert({ "_id":name, "name":name, "Category":predval,"date":str(datetime.now()),"Summary":summary})
			except IndexError:
				print('Text Not found in the news article.')
				#todosnew_l={"Category":name+": invalid News Article URL"}
				return redirect("/invalid")				
		else:
			temp=todos.find({'_id':name})
			print(temp)
			for i in temp:
				print(i)
				predval=i['Category']
			print(predval)
		todosnew_l={'Category':predval}
	elif option =='Batch':
		todosnew_l={'Category':'Batch Mode Not available at this moment'}
	todos_l = todos.find().sort("date",-1)
	return render_template('index.html',a1=a1,todos=todos_l,todos_new=todosnew_l,t=title,h=heading)

#Handle Retrain Model.
@app.route("/retrain")
def retrainmodel():
	a1="active"
	#lasttraineddate=retraincol.date
	#print(lasttraineddate)
	#cursor = db['predictionresults'].find({'date':{'$gte':lasttraineddate}})
	#print(cursor.count())
	retrain_model()
	retraincol.insert({ 'lasttraineddate':str(datetime.now()) })
	todosnew_l={'Category':'Retrained model successfully.'}
	todos_l = todos.find()
	return render_template('index.html',a1=a1,todos=todos_l,todos_new=todosnew_l,t=title,h=heading)

#Handle model training
@app.route("/train")
def trainmodel():
	a1="active"
	df=read_from_csv("Data/train.csv")
	processed_df = process_train_data(df)
	train_model(processed_df)
	todosnew_l={'Category':'Trained model successfully.'}
	todos_l = todos.find()
	return render_template('index.html',a1=a1,todos=todos_l,todos_new=todosnew_l,t=title,h=heading)

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)
