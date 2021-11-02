from pyspark.sql import SparkSession,SQLContext
from pyspark import SparkConf, SparkContext
import json

class MongoConnector:
   def __init__(self):
     self.my_spark=None
     self.df=None
     self.uri='mongodb://root:example@localhost:27017/newsclassifier.producer_collection_newsclassifier'
   
   def init_sparksession(self):
     self.uri = 'mongodb://root:example@localhost:27017/newsclassifier.producer_collection_newsclassifier'
     print(f'the init sparksession {self.uri}')
     self.my_spark = SparkSession \
                     .builder \
                     .appName("ProcessrUtility") \
                     .master("spark://spark-master:7077") \
                     .config("spark.executor.memory", "1g") \
                     .config("spark.mongodb.input.uri",self.uri) \
                     .config("spark.mongodb.output.uri",self.uri) \
                     .config("spark.jars.packages","org.mongodb.spark:mongo-spark-connector_2.12:2.4.2") \
                     .getOrCreate()
     print(f'the spark session is {self.my_spark}')
     return self.my_spark
      
     
   def read(self,my_spark):
     print(f' the read usri is {self.uri}')
     self.df=my_spark.read.format("mongo").load()
     # self.df=my_spark.read.format("mongo").option("uri",self.uri).load()
     print(f' the schema is {self.df.printSchema()}')
     #self.df.printSchema()
     return self.df
     
     
     
# .option("spark.mongodb.input.uri","mongodb://root:example@mongo:27017/producer_collection_newsclassifier")
#spark.read.format("com.mongodb.spark.sql.DefaultSource").load()
