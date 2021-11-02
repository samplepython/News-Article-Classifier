import os
import uvicorn
import requests
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from ProcessrUtility import process_data

TRAINR_ENDPOINT = os.getenv("TRAINR_ENDPOINT")

# defining the main app
app = FastAPI(title="processr", docs_url="/")

# class which is expected in the payload while training
class DataIn(BaseModel):
    title: str
    datetime: str
    summary: str    
    source:str 
    topic: str


# Route definitions
@app.get("/ping")
# Healthcheck route to ensure that the API is up and running
def ping():
    return {"ping": "pong"}


@app.post("/process", status_code=200)
# Route to take in data from MongoDB, process it and send it for training.
def process():   
    (train,test) = process_data()
    # send the processed data to trainr for training
    response = requests.post(f"{TRAINR_ENDPOINT}/train", json=train)
    #response = requests.post(f"{TRAINR_ENDPOINT}/train", json=train)
    return {"detail": "Processing successful"}


# Main function to start the app when main.py is called
if __name__ == "__main__":
    # Uvicorn is used to run the server and listen for incoming API requests on 0.0.0.0:8888
    uvicorn.run("main:app", host="0.0.0.0", port=8888, reload=True)
