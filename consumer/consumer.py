from kafka import KafkaConsumer
from json import loads
from time import sleep
from pathlib import Path
import os
import sys


def begin():

    sleep(15)
    BROKER = os.getenv('BROKER', 'localhost:9092')
    # TOPICS = ['news','tech']
    TOPIC = 'news'
    sleep(15)
    
    # consumer_news = KafkaConsumer(
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=[BROKER],
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='my-group')
        
    # consumer_tech = KafkaConsumer(
    #     TOPICS[1],
    #     bootstrap_servers=[BROKER],
    #     auto_offset_reset='earliest',
    #     enable_auto_commit=True,
    #     group_id='my-group2')

    mongo_url = os.getenv('ME_CONFIG_MONGODB_URL', 'mongodb://root:example@mongo:27017/')
    db_name = 'newsclassifier'
    collection_name = 'consumer_collection_newsclassifier'
    # establishing the connection
    mydb = mongo_db_connect(mongo_url, db_name)
    mycol=mongo_db_create_collection(mydb, collection_name)

    for article in consumer:
        # print(f'article in consumer news is:{article}')
        article = article.value         
        article = loads(article) 
        mongo_db_insert(mycol, article)

if __name__ == '__main__':
    current_working_dir = str(Path.cwd())
    utils_path = current_working_dir+'/common_utils'
    sys.path.insert(0, utils_path)
    from db_utils import mongo_db_connect, mongo_db_create_collection, mongo_db_insert
    # from threading import Thread
    # import time
 
    begin()
