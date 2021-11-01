# News-Article-Classifier
The project have three docker containers:
1. producer
2. consumer
3. processr

Data Ingestion:

producer and consumer containers are used for data ingestion. These will use the below predefined docker images from the docker hub:
  1. zookeeper
  2. kafka
  3. spark-master
  4. spark-worker
  
   As part of data ingestion, we have used two sources of data namely:
     1. Freenews API
     2. BBC RSS Feed
      
   The data keywords are provided for the Freenews API for fetching the data about the keywords.  The returned data from the freenews api is then parsed to fetch the below fields:
    title<br/>
	date/ time<br/>
	summary<br/>
	topic/ category<br/>
	source<br/>
	We are streaming the data to the consumer using the Kafka broker.
	From the consumer side we are extracting the data from the Kafka broker and pushing the same into MongoDB.
     
    The two docker projects producer and consumer will be run as part of Data Ingestion.

Data Pre-Processing, Model Training and Prediction:

   As part of building the model we have done in the below way:
     1. Took the existing new classifier data set from kaggle.
     2. Processed the data set and retrieved the required features from it. 
     3. Divided the data into Test and Train Data set. We took around 50k data records to train the model.
     4. Trained the model with this Data.
     5. Done predictions with test data and measured the accuracy.
     6. Save the Model.
     7. We then fetched the data from Mongo DB which was filled by the Data Ingestion service.
     8. Loaded the already saved model.
     9. Re trained the model with the collected data.
     10. And then saved it locally.
 
  Then we tested the Predictions with Fast API. 
  
  After some predictions, we retrained the model using the above steps 7 to 10.
  
  Flask Web UI is provided to user to enter the news article URL for prediction.
  
Installation How-to?

    1. Every service is embedded as a docker image.  We have the corresponding Docker file and the requirements.txt for each   project separately.
	2. Also, we have the docker-compose.yaml file to build the source and making the services up.
	3. A pre-requisite is to have the docker and docker-compose installed on the machine or VM.
	4. The current implementation was only tested on Ubuntu 20.04.
	
Below is the highlevel architecture diagram of the application:
![Architecture](https://github.com/samplepython/News-Article-Classifier/blob/main/documents/Week4/DataIngestion.drawio/NewClassifierArchitecture.png)

Below are the sample sequence diagrams:
Training Sequence Diagram:
![Training](https://github.com/samplepython/News-Article-Classifier/blob/main/documents/Week4/Sequence/Sequence-Diagrams/training-news-sequence-diagram.png)
Prediction Sequence Diagram:
![Predict_news](https://github.com/samplepython/News-Article-Classifier/blob/main/documents/Week4/Sequence/Sequence-Diagrams/predict-news-sequence-diagram.png)

Below are some of the sample output screens:


