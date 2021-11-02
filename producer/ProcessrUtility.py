import os
import sys
from pathlib import Path
import sweetviz as sv
import pickle
import SparkConnector
from pyspark import SparkConf, SparkContext
from sklearn.naive_bayes import GaussianNB
from pyspark.ml import Pipeline 
from pyspark.ml.feature import CountVectorizer,StringIndexer,RegexTokenizer,StopWordsRemover
from pyspark.sql.functions import col, udf,regexp_replace,isnull 
from pyspark.sql.types import StringType,IntegerType
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


#load the data from MongoDB
def read_from_db():
    mongo_connector = SparkConnector.MongoConnector()
    print(f'spark connector{mongo_connector}')
    ss = mongo_connector.init_sparksession()
    print(f' after init spark session {ss}')
    return mongo_connector.read(ss)

#analyze for Missing Values using sweetviz
def analyze_missing_values(data):    
    return sv.analyze(data)

# check null values in TITLE and topic columns    
def null_value_count(df):
  null_columns_counts = []
  numRows = df.count()
  for k in df.columns:
    nullRows = df.where(col(k).isNull()).count()
    if(nullRows > 0):
      temp = k,nullRows
      null_columns_counts.append(temp)
  return(null_columns_counts)

# function to process data and return it in correct format
def process_data():
    data = read_from_db()
    #report = analyze_missing_values(data)
    #report.show_notebook()
    title_category = data.select("title","summary","topic")
    title_category.show()
    null_columns_count_list = null_value_count(title_category)
    #spark.createDataFrame(null_columns_count_list, ['Column_With_Null_Value','Null_Values_Count']).show()
    
    #Drop the null values
    title_category = title_category.dropna()
    title_category.count()
    title_category.show(truncate=False)
    
    #List Top 20 news categories
    title_category.groupBy("topic").count().orderBy(col("count").desc()).show(truncate=False)
    
    #Top 20 news title
    title_category.groupBy("title").count().orderBy(col("count").desc()).show(truncate=False)
    
    #Removing numbers from titles
    title_category = title_category.withColumn("only_str",regexp_replace(col('title'), '\d+', ''))
    title_category.select("title","only_str").show(truncate=False)
    
    #Spliting text into words
    regex_tokenizer = RegexTokenizer(inputCol="only_str", outputCol="words", pattern="\\W")
    raw_words = regex_tokenizer.transform(title_category)
    raw_words.show()
    
    #Removing stop words
    remover = StopWordsRemover(inputCol="words", outputCol="filtered") 
    words_df = remover.transform(raw_words)
    words_df.select("words","filtered").show()
    
    #encode column of category to a column of category indices
    indexer = StringIndexer(inputCol="topic", outputCol="categoryIndex") 
    feature_data = indexer.fit(words_df).transform(words_df)
    feature_data.select("topic","categoryIndex").show()
    
    #Converting text into vectors of token counts.
    cv = CountVectorizer(inputCol="filtered", outputCol="features") 
    model = cv.fit(feature_data)
    countVectorizer_features = model.transform(feature_data)
    
    #Partition Training & Test sets
    (trainingData, testData) = countVectorizer_features.randomSplit([0.8,0.2],seed = 11)

    return (trainingData,testData)

if __name__ == '__main__':
    current_working_dir = str(Path.cwd())
    utils_path = current_working_dir+'/Utilities'
    sys.path.insert(0, utils_path)
    import SparkConnector 
    process_data()
