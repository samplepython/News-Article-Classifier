#!/usr/bin/python3

from kafka import KafkaProducer
#from random import randint
#import pickle
from pathlib import Path
from time import sleep
import json
from bson import json_util
import os
import sys
import requests
from copy import deepcopy
from bs4 import BeautifulSoup

def rssfeed():
    data={}
    news_items = []
    url = "http://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml"
    resp = requests.get(url)
    soup = BeautifulSoup(resp.content, features="xml")
    print(soup.prettify())    
    items = soup.findAll('item')    
    print(items)
    print(len(items))
    for item in items:
      news_item = {}
      news_item['title'] = item.title.text
      news_item['summary'] = item.description.text
      news_item['link'] = item.link.text
      news_item['published_date'] = item.pubDate.text
      news_item['topic']='news'
      news_items.append(news_item)
    data["articles"]=news_items
    data["page_size"]=len(news_items)
    return data
    
def rapidapi_response(input_params):
    response = requests.get(f"https://free-news.p.rapidapi.com/v1/search",
                            params=input_params,
                            headers={
                                "X-RapidAPI-Host": "free-news.p.rapidapi.com",
                                "X-RapidAPI-Key": "eee47c64fbmsh20b8f3b65036c54p12e1f5jsn8af4f3266d93"
                                    }
                            )
    data = response.json()
    return data

def begin():
    sleep(15)
    BROKER = os.getenv('BROKER', 'localhost:9092')
    TOPICS = ['news','tech','business']
    sleep(15)
    
    try:
        producer = KafkaProducer(bootstrap_servers=BROKER)
    except Exception as e:
        print(f"ERROR --> {e}")
        sys.exit(1)

# Mongo data handling
    mongo_url = os.getenv('ME_CONFIG_MONGODB_URL', 'mongodb://root:example@mongo:27017/')
    db_name = 'newsclassifier'
    collection_name = 'producer_collection_newsclassifier'
    # establishing the connection
    mydb = mongo_db_connect(mongo_url, db_name)
    mycol=mongo_db_create_collection(mydb, collection_name)
    
# # Mysql data handling    
#     database_name = os.getenv('MYSQL_DATABASE', 'newsclassifier')
#     user_name = os.getenv('MYSQL_USER', 'testuser')
#     password = os.getenv('MYSQL_PASSWORD', 'testpassword')
#     host_name = 'mysql-db'
#     # establishing the connection
#     my_conn = mysql_db_connect(host_name, database_name, user_name, password)
#     my_table = 'tbl_newsclassifier'
#     table_name = mysql_db_create_table(my_conn, my_table)
    
    count = 10
    while count > 0:
        input_params = {"q": 'Elon Musk', "lang": 'en'}
        api_response_data = rapidapi_response(input_params)
        #if 'page_size' not in api_response_data.keys():
          #api_response_data = rssfeed() 
        for article in api_response_data['articles']:
            article = {"title":article['title'],
                       "published_date":article['published_date'],
                       "summary":article['summary'],
                       "topic":article['topic'],
                       "source":article['link']
                       }
            mongo_document = deepcopy(article)
            mongo_db_insert(mycol, mongo_document)  
            # mysql_record = (article['title'], article['published_date'],
            #                 article['summary'], article['topic'], article['clean_url'])
            # mysql_db_insert(my_conn, table_name, mysql_record)   
            if article["topic"] in TOPICS:
                producer.send(article["topic"],json.dumps(article, default=json_util.default).encode('utf-8'))
                # print(f'the topic is:{article["topic"]}')
        count -= 1
        if count == 0:
            break
        else:
            sleep(60)



if __name__ == '__main__':
    current_working_dir = str(Path.cwd())
    utils_path = current_working_dir+'/common_utils'
    sys.path.insert(0, utils_path)
    from db_utils import mongo_db_connect, mongo_db_create_collection, mongo_db_insert
    from db_utils import mysql_db_connect, mysql_db_create_table, mysql_db_insert
    begin()
