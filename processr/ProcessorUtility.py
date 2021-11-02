import re
import string
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import Row
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag
import langid


#check for blank spaces in a string provided
def check_blanks(data_str):
    is_blank = str(data_str.isspace())
    return is_blank

# remove non ASCII characters
def strip_non_ascii(data_str):
    ''' Returns the string without non ASCII characters'''
    stripped = (c for c in data_str if 0 < ord(c) < 127)
    return ''.join(stripped)

# classify the language of the text
def check_lang(data_str):
    predict_lang = langid.classify(data_str)
    if predict_lang[1] >= .9:
        language = predict_lang[0]
    else:
        language = 'NA'
    return language

# remove some unwanted characters/text from the string provided using regex
def remove_features(data_str):
    # compile regex
    url_re = re.compile('https?://(www.)?\w+\.\w+(/\w+)*/?')
    punc_re = re.compile('[%s]' % re.escape(string.punctuation))
    num_re = re.compile('(\\d+)')
    mention_re = re.compile('@(\w+)')
    alpha_num_re = re.compile("^[a-z0-9_.]+$")
    # convert to lowercase
    data_str = data_str.lower()
    # remove hyperlinks
    data_str = url_re.sub(' ', data_str)
    # remove @mentions
    data_str = mention_re.sub(' ', data_str)
    # remove puncuation
    data_str = punc_re.sub(' ', data_str)
    # remove numeric 'words'
    data_str = num_re.sub(' ', data_str)
    # remove non a-z 0-9 characters and words shorter than 3 characters
    list_pos = 0
    cleaned_str = ''
    for word in data_str.split():
        if list_pos == 0:
            if alpha_num_re.match(word) and len(word) > 2:
                cleaned_str = word
            else:
                cleaned_str = ' '
        else:
            if alpha_num_re.match(word) and len(word) > 2:
                cleaned_str = cleaned_str + ' ' + word
            else:
                cleaned_str += ' '
        list_pos += 1
    return cleaned_str

#stop words removal
def remove_stop_words(data_str):
    # expects a string
    stops = set(stopwords.words("english"))
    list_pos = 0
    cleaned_str = ''
    text = data_str.split()
    for word in text:
        if word not in stops:
            # rebuild cleaned_str
            if list_pos == 0:
                cleaned_str = word
            else:
                cleaned_str = cleaned_str + ' ' + word
            list_pos += 1
    return cleaned_str

# remove the tags
def tag_and_remove(data_str):
    cleaned_str = ' '
    # noun tags
    nn_tags = ['NN', 'NNP', 'NNP', 'NNPS', 'NNS']
    # adjectives
    jj_tags = ['JJ', 'JJR', 'JJS']
    # verbs
    vb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    nltk_tags = nn_tags + jj_tags + vb_tags
    # break string into 'words'
    text = data_str.split()
    # tag the text and keep only those with the right tags
    tagged_text = pos_tag(text)
    for tagged_word in tagged_text:
        if tagged_word[1] in nltk_tags:
            cleaned_str += tagged_word[0] + ' '
    return cleaned_str

#do lemmatization
def lemmatize(data_str):
    # expects a string
    list_pos = 0
    cleaned_str = ''
    lmtzr = WordNetLemmatizer()
    text = data_str.split()
    tagged_words = pos_tag(text)
    for word in tagged_words:
        if 'v' in word[1].lower():
            lemma = lmtzr.lemmatize(word[0], pos='v')
        else:
            lemma = lmtzr.lemmatize(word[0], pos='n')
        if list_pos == 0:
            cleaned_str = lemma
        else:
            cleaned_str = cleaned_str + ' ' + lemma
        list_pos += 1
    return cleaned_str
    
#function to process the training data
def process_train_data(df):
    ''' function to process data for text normalization'''  
    # drop the Title column, that are not required
    df = df.drop('title')  
    
    articles_df = df.select("summary","topic")
    articles_df.show(5)

    raw_cols = articles_df.columns
    # Register all the functions in pyspark udf with Spark Context
    check_lang_udf = udf(check_lang, StringType())
    strip_non_ascii_udf = udf(strip_non_ascii, StringType())
    remove_stop_words_udf = udf(remove_stop_words, StringType())
    remove_features_udf = udf(remove_features, StringType())
    tag_and_remove_udf = udf(tag_and_remove, StringType())
    lemmatize_udf = udf(lemmatize, StringType())
    check_blanks_udf = udf(check_blanks, StringType())    
    
    lang_summary_df = articles_df.withColumn("lang", check_lang_udf(articles_df["summary"]))
    en_summary_df = lang_summary_df.filter(lang_summary_df["lang"] == "en")
    en_summary_df.show(4)
    
    non_asci_summary_df = articles_df.withColumn('non_asci_summary',strip_non_ascii_udf(articles_df['summary']))
    non_asci_summary_df.show(4,True) 
    
    rm_stops_summary_df = non_asci_summary_df.select(raw_cols)\
                                .withColumn("stop_words_summary", remove_stop_words_udf(non_asci_summary_df["summary"]))
    rm_stops_summary_df.show(4)
 
    rm_features_summary_df = rm_stops_summary_df.select(raw_cols+["stop_words_summary"])\
                                .withColumn("feature_summary", remove_features_udf(rm_stops_summary_df["stop_words_summary"]))
    rm_features_summary_df.show(4)
  
    tagged_summary_df = rm_features_summary_df.select(raw_cols+["feature_summary"]) \
                                .withColumn("tagged_summary", tag_and_remove_udf(rm_features_summary_df.feature_summary))
    tagged_summary_df.show(4)  
    
    lemm_summary_df = tagged_summary_df.select(raw_cols+["tagged_summary"]) \
                                .withColumn("lemm_summary", lemmatize_udf(tagged_summary_df["tagged_summary"]))
    lemm_summary_df.show(4)
    
    check_blanks_summary_df = lemm_summary_df.select(raw_cols+["lemm_summary"])\
                                .withColumn("is_blank", check_blanks_udf(lemm_summary_df["lemm_summary"]))
    # remove blanks
    no_blanks_summary_df = check_blanks_summary_df.filter(check_blanks_summary_df["is_blank"] == "False")
    
    no_blanks_summary_df = no_blanks_summary_df.drop('summary')
    # drop duplicates
    no_duplicate_summary_df = no_blanks_summary_df.dropDuplicates(['lemm_summary', 'topic'])

    final_df = no_duplicate_summary_df.select('lemm_summary','topic')
    final_df.show(4)
   
    # Encode column of topic to a column of category indices
    print('encode column of Topic to a column of category indices')
    indexer = StringIndexer(inputCol="topic", outputCol="label") 
    final_df = indexer.fit(final_df).transform(final_df)
    final_df.groupBy(['topic']).agg({'label': 'avg'}).show()
    
    return final_df
    
#process prediction summary string data
def process_predict_data_str(strSummary):
    my_spark = SparkSession.\
        builder.\
        appName("PredictionUtility").\
        config("spark.jars.packages","org.mongodb.spark:mongo-spark-connector_2.12:3.0.1").\
        getOrCreate()
    schema = StructType([StructField('summary', StringType())])
    rows = [Row(summary=strSummary)]
    dfsummary = my_spark.createDataFrame(rows, schema)
    return process_predict_data(dfsummary)

#process prediction summary dataframe
def process_predict_data(df):
    ''' function to process data for text normalization'''
    # print(df.columns)    
    articles_df = df.select("summary")
    articles_df.show(5)
    raw_cols = articles_df.columns
    # Register all the functions in pyspark udf with Spark Context
    check_lang_udf = udf(check_lang, StringType())
    strip_non_ascii_udf = udf(strip_non_ascii, StringType())
    remove_stop_words_udf = udf(remove_stop_words, StringType())
    remove_features_udf = udf(remove_features, StringType())
    tag_and_remove_udf = udf(tag_and_remove, StringType())
    lemmatize_udf = udf(lemmatize, StringType())
    check_blanks_udf = udf(check_blanks, StringType())    
    
    lang_summary_df = articles_df.withColumn("lang", check_lang_udf(articles_df["summary"]))
    en_summary_df = lang_summary_df.filter(lang_summary_df["lang"] == "en")
    en_summary_df.show(4)
    
    non_asci_summary_df = articles_df.withColumn('non_asci_summary',strip_non_ascii_udf(articles_df['summary']))
    non_asci_summary_df.show(4,True) 
    
    rm_stops_summary_df = non_asci_summary_df.select(raw_cols)\
                                .withColumn("stop_words_summary", remove_stop_words_udf(non_asci_summary_df["summary"]))
    rm_stops_summary_df.show(4)
 
    rm_features_summary_df = rm_stops_summary_df.select(raw_cols+["stop_words_summary"])\
                                .withColumn("feature_summary", remove_features_udf(rm_stops_summary_df["stop_words_summary"]))
    rm_features_summary_df.show(4)
  
    tagged_summary_df = rm_features_summary_df.select(raw_cols+["feature_summary"]) \
                                .withColumn("tagged_summary", tag_and_remove_udf(rm_features_summary_df.feature_summary))
    tagged_summary_df.show(4)  
    
    lemm_summary_df = tagged_summary_df.select(raw_cols+["tagged_summary"]) \
                                .withColumn("lemm_summary", lemmatize_udf(tagged_summary_df["tagged_summary"]))
    lemm_summary_df.show(4)
    
    check_blanks_summary_df = lemm_summary_df.select(raw_cols+["lemm_summary"])\
                                .withColumn("is_blank", check_blanks_udf(lemm_summary_df["lemm_summary"]))
    # remove blanks
    no_blanks_summary_df = check_blanks_summary_df.filter(check_blanks_summary_df["is_blank"] == "False")
    
    no_blanks_summary_df = no_blanks_summary_df.drop('summary')
    # drop duplicates
    no_duplicate_summary_df = no_blanks_summary_df.dropDuplicates(['lemm_summary'])

    final_df = no_duplicate_summary_df.select('lemm_summary')
    final_df.show(4)
    return final_df

#load the data from MongoDB
# def read_from_db():  
    
#      my_spark = SparkSession.\
# 	     builder.\
# 	     appName("ProcessrUtility").\
# 	     config("spark.mongodb.input.uri","mongodb://root:example@mongo:27017/newsclassifier.producer_collection_newsclassifier?authSource=admin").\
# 	     config("spark.mongodb.output.uri","mongodb://root:example@mongo:27017/newsclassifier.producer_collection_newsclassifier?authSource=admin").\
# 	     config("spark.jars.packages","org.mongodb.spark:mongo-spark-connector_2.12:3.0.1").\
# 	     getOrCreate()         
    
     #df=my_spark.read.format("com.mongodb.spark.sql.DefaultSource").option("uri", "mongodb://root:example@mongo:27017/newsclassifier.producer_collection_newsclassifier").load()
     #df=my_spark.read.format("com.mongodb.spark.sql.DefaultSource").option("database", "newsclassifier").option("collection", "producer_collection_newsclassifier").load()
     #print('in read printschma')     
     #df.show()
     #df = my_spark.read.format("mongo").load()
    #  df = my_spark.read.format("com.mongodb.spark.sql.DefaultSource").option("database","newsclassifier").option("collection", "producer_collection_newsclassifier").load()
    #  df.printSchema()
    # return my_spark,df

if __name__ == '__main__':
    pass
