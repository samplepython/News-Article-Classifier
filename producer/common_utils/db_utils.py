#!/usr/bin/python3

from pymongo import MongoClient
from mysql import connector

# Mongo Utils
def mongo_db_connect(url, db_name):
    # establishing the connection
    myclient = MongoClient(url)
    # mydb = myclient["newsclassifier"]
    mydb = myclient[db_name]    
    return mydb

def mongo_db_create_collection(mydb, collection_name):  
    # mycol = mydb["collection_newsclassifier"]	
    mycol = mydb[collection_name]	
    return mycol

def mongo_db_insert(collection, document): 
    # print(f'the document is: {document}')
    collection_obj = collection.insert_one(document)
    return collection_obj


# Mysql Utils
def mysql_db_connect(host_name, db_name, user_name, password):
     conn = connector.connect(user=user_name, password=password,
                                 host=host_name, database=db_name)
     # Setting auto commit false
     conn.autocommit = True
     return conn

def mysql_db_create_table(conn, table_name):
    # Creating a cursor object using the cursor() method
    mycursor = conn.cursor()
    mycursor.execute("SHOW TABLES")
    is_table_exists = False
    for check_table_name in mycursor:
        # print(f' the existing table name is : {str(table_name)}')
        if check_table_name[0] == table_name:
            # print(f'table name {table_name} already exists')
            # Dropping tbl_newsclassifier table if already exists.
            # sql = "DROP TABLE IF EXISTS " + table_name
            # mycursor.execute(sql)
            # Creating table
            sql = ''' CREATE TABLE ''' +  table_name + '''(
                                                title text, 
                                                datetime datetime, 
                                                summary longtext, 
                                                topic text, 
                                                source text
                                                )'''
            mycursor.execute(sql)
            is_table_exists = True
            break
    if not is_table_exists:
        sql = ''' CREATE TABLE ''' + table_name + '''(
                                            title text, 
                                            datetime datetime, 
                                            summary longtext, 
                                            topic text, 
                                            source text
                                            )'''
        mycursor.execute(sql)
    return table_name

def mysql_db_insert(conn, table_name, record):
    mycursor = conn.cursor()
    sql = "INSERT INTO  " + table_name + " VALUES (%s, %s, %s, %s, %s)"
    mycursor.execute(sql, record)
    # print(f' the row in the database is {record}')

if __name__ == '__main__':
    pass
