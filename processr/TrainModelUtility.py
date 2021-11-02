from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml import Pipeline,PipelineModel
from pyspark.ml.classification import NaiveBayes, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from ProcessorUtility import process_train_data
import mlflow
import mlflow.mleap

def mlflow_fit_model(pipeline, hashingTF, trainingData):
  '''Start a new MLflow run'''
  with mlflow.start_run() as run:
    # Fit the model, performing cross validation to improve accuracy
    paramGrid = ParamGridBuilder().addGrid(hashingTF.numFeatures, [1000, 2000]).build()
    cv = CrossValidator(estimator=pipeline, evaluator=MulticlassClassificationEvaluator(), estimatorParamMaps=paramGrid)
    # use Mlflow to log the best model into Mleap format
    # extract the optimal Pipeline model to be logged
    cvModel = cv.fit(trainingData)
    model = cvModel.bestModel

    # Log the model within the MLflow run
    mlflow.mleap.log_model(spark_model=model, sample_input=trainingData, artifact_path="models")


# Function for Displaying the Confusion Matrix
# This function displays the confusion matrix of the model
def confusion_mat(color, y_true, y_pred, model_name):
    print(classification_report(y_true, y_pred))
    cof=confusion_matrix(y_true, y_pred)
    cof=pd.DataFrame(cof, index=[i for i in range(1,5)], columns=[i for i in range(1,5)])
    sns.set(font_scale=1.5)
    plt.figure(figsize=(8,8));

    sns.heatmap(cof, cmap=color,linewidths=1, annot=True,square=True, fmt='d', cbar=False,xticklabels=['World','Sports','Business','Science'],yticklabels=['World','Sports','Business','Science']);
    plt.xlabel("Predicted Classes");
    plt.ylabel("Actual Classes");
    plt.savefig("models/"+str(model_name)+"_confusion_metrix.png")

# function to load the saved model from storage
def load_model(modelobj, model_name):
    global nb_model
    nb_model = PipelineModel.load("models/"+str(model_name)+"_newsclassifier.model")
    return nb_model

#function to save the trained/retrained model on to a specified path.
def save_model(model_obj, model_name):
    # save the model
    model_obj.write().overwrite().save("models/"+str(model_name)+"_newsclassifier.model")

# Configure an ML pipeline, which consists of tree stages: tokenizer, hashingTF, and nb.
def model_pipeline(classifier_obj):     
    tokenizer = Tokenizer(inputCol="lemm_summary", outputCol="words")
    hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="rawFeatures")
    # vectorizer = CountVectorizer(inputCol= "words", outputCol="rawFeatures")
    idf = IDF(minDocFreq=3, inputCol=hashingTF.getOutputCol(), outputCol="features")    
    pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, classifier_obj])
    return pipeline

#function to retrain the model on demand.
def retrain_model_train(df):
     # Naive Bayes Model
    nb_clf = NaiveBayes()    
    nb_model = load_model(nb_clf, "NaiveBayes")
    df = df.withColumnRenamed("Category", "topic").withColumnRenamed("Summary","summary")     
    processed_df=process_train_data(df)
    nb_pipeline = model_pipeline(nb_model)
    # Train model. This also runs the indexers.
    nb_new_model = nb_pipeline.fit(processed_df)   
    save_model(nb_new_model, "NaiveBayes1")   
    print("retraining model completed.")

# function to train and save the model as part of the feedback loop
def train_model(processed_df):    
    # Partition Training & Test sets
    print('Partition Training & Test sets.')
    (trainingData, testData) = processed_df.randomSplit([0.8,0.2],seed = 11)
    print("Training data set")
    trainingData.show(4)
    print("Test Data Set")
    testData.show(4)     
    # Naive Bayes Model
    nb_clf = NaiveBayes()
    # Pipeline Architecture
    nb_pipeline = model_pipeline(nb_clf)
    # Train model. This also runs the indexers.
    nb_model = nb_pipeline.fit(trainingData)
    nb_predictions = nb_model.transform(testData)
    # Select example rows to display.
    nb_predictions.select("lemm_summary", "label", "prediction").show(4,False)
    nb_evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    nb_accuracy = nb_evaluator.evaluate(nb_predictions)
    print(f'===Naive Bayes Metrics:===')
    print("Accuracy of NaiveBayes is = %g"% (nb_accuracy)) 
    print("Test Error of NaiveBayes = %g " % (1.0 - nb_accuracy))
    save_model(nb_model, "NaiveBayes")
    y_true = nb_predictions.select(['label']).collect()
    y_pred = nb_predictions.select(['prediction']).collect()
    confusion_mat('Greens', y_true, y_pred, "NaiveBayes")
    # mlflow_fit_model(nb_pipeline, hashingTF, trainingData)
    
    # RandomForestClassifier Model
    # rf_clf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label', numTrees=500)
    # rf_pipeline = model_pipeline(rf_clf)
    # rf_model = rf_pipeline.fit(trainingData)
    # rf_predictions = rf_model.transform(testData)
    # # Select example rows to display.
    # rf_predictions.select("lemm_summary", "label", "prediction").show(4,False)
    # rf_evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    # rf_accuracy = rf_evaluator.evaluate(rf_predictions)
    # print(f'===RandomForestClassifier Metrics:===')
    # print("Accuracy of RandomForestClassifier is = %g"% (rf_accuracy)) 
    # print("Test Error of RandomForestClassifier = %g " % (1.0 - rf_accuracy))
    # save_model(rf_model, "RandomForestClassifier")
    # y_true = rf_predictionstransform.select(['label']).collect()
    # y_pred = rf_predictions.select(['prediction']).collect()
    # confusion_mat('Blues', y_true, y_pred, "RandomForestClassifier")
    

    print("Training Model Completed Successfully.")

if __name__ == '__main__':
    pass
