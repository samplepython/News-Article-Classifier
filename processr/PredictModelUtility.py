import requests
from bs4 import BeautifulSoup
from pyspark.ml.feature import CountVectorizer,RegexTokenizer,StopWordsRemover,IDF
from pyspark.sql.functions import col,regexp_replace 
from pyspark.sql.types import StringType
from pyspark.ml.classification import NaiveBayes
from pyspark.sql import SparkSession
from pyspark.sql.functions import lower, col
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import Row
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml import PipelineModel, Pipeline
from TrainModelUtility import retrain_model_train


# define the class encodings and reverse encodings
classes = {0:"Business", 1:"Sports", 2:"Science", 3:"World"}
r_classes = {y: x for x, y in classes.items()}

def init_spark_session():
    global my_spark
    my_spark = SparkSession.\
	     builder.\
	     appName("ProcessrUtility").\
	     config("spark.mongodb.input.uri","mongodb://root:example@mongo:27017/newsclassifier.predictionresults?authSource=admin").\
	     config("spark.mongodb.output.uri","mongodb://root:example@mongo:27017/newsclassifier.predictionresults?authSource=admin").\
	     config("spark.jars.packages","org.mongodb.spark:mongo-spark-connector_2.12:3.0.1").\
	     getOrCreate()

# function to load the model
def load_model(model_obj, model_name):
    global nb_model     
    nb_model = PipelineModel.load("models/"+str(model_name)+"_newsclassifier.model")
    return nb_model

# model     
def model_pipeline(classifier_obj):
     # Configure an ML pipeline, which consists of tree stages: tokenizer, hashingTF, and nb.
    tokenizer = Tokenizer(inputCol="lemm_summary", outputCol="words")
    hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="rawFeatures")
    # vectorizer = CountVectorizer(inputCol= "words", outputCol="rawFeatures")
    idf = IDF(minDocFreq=3, inputCol=hashingTF.getOutputCol(), outputCol="features")    
    pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, classifier_obj])
    return pipeline

#retrain Model
def retrain_model():   
    df = my_spark.read.format("com.mongodb.spark.sql.DefaultSource").option("database","newsclassifier").option("collection", "predictionresults").load()
    df.printSchema()
    retrain_model_train(df) 

# function to predict the flower using the model
def predict(data_df):
    # Naive Bayes Model
    nb_clf = NaiveBayes()
    # Load Model
    nb_model = load_model(nb_clf, "NaiveBayes")  
    # predict value 
    nb_predictions = nb_model.transform(data_df)
    print(f'nb_predictions is:')
    nb_predictions.show()
    y_pred = nb_predictions.select(['prediction']).collect()    
    return classes[int(y_pred[0][0])]

#This function will filterout all the non standard news articles and allow only valid urls to proceed
#further for processing and prediction
def fakenewsurlidentifier(url):  
    data={}  
    source = "http://newspaper-demo.herokuapp.com/articles/show?url_to_clean="+url
    resp = requests.get(source)
    soup = BeautifulSoup(resp.content,  "html.parser")
    table = soup.find_all("table", class_="table")[0]
    text=""
    new_table = {}
    row_marker=0
    for row in table.find_all('tr'):
        column_marker = 0
        columns = row.find_all('td')
        for column in columns:
            coltext=column.get_text()
            print(coltext)
            if len(new_table.keys()) > 0:
                column_marker += 1
                new_table['Text']=column.get_text()
                break
            if coltext == 'Text':
                new_table['Text'] = ""
            
        if len(new_table.keys()) > 0:
            break     
    return new_table

 #Process the URL data provided and get the content of it.   
def ProcessUrl(query_data):
    init_spark_session()
    data={}
    mystr =""
    url =  query_data.dict()["url"]
    if len(url) > 0:
        print("URL Received: "+url)   
        data = fakenewsurlidentifier(url)
        print(data)
        mystr = data['Text']    
        print(f'the  summary in single line is:{mystr}')    
    return mystr