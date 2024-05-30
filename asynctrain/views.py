import asyncio
import json
import os
from django.http import JsonResponse
import pandas as pd
from ml_model.models.linear_regression import LumbaLinearRegression
from ml_model.models.decision_tree import LumbaDecisionTreeClassifier
from ml_model.models.random_forest import LumbaRandomForestRegressor
from ml_model.models.kmeans import LumbaKMeans
from ml_model.models.arima import LumbaARIMA
from ml_model.models.lstm import LumbaLSTM
import requests
import joblib
from json import dumps
import numpy as np
from modeling import settings
import time
from .obseg_views import *
import io
import zipfile

async def asyncforecastingtrain(df, training_record, model_forecasting_metadata):  

    # update training record to 'in progress'
    url = f'http://{settings.BACKEND_SERVICE_INTERNAL_IP}:{settings.BACKEND_SERVICE_RUNNING_PORT}/forecasting/updateforecastingrecord/'
    json = {'id': training_record['id'], 'status':'in progress'}
    record = requests.post(url, json=json)
    print("training with record id "+ str(record.json()['id']) + " in progress")

    if model_forecasting_metadata['algorithm'] == 'ARIMA':
      LA = LumbaARIMA(df)
      response = LA.train_model(train_column_name=model_forecasting_metadata['target'])
      metrics_scores = {"r2_score":response['r2_score'],
                        "mean_squared_error":response['mean_squared_error'],
                        "root_mean_squared_error":response['root_mean_squared_error'],}
      model_forecasting_metadata["metrics_scores"] = dumps(metrics_scores)
    if model_forecasting_metadata['algorithm'] == 'LSTM':
      LL = LumbaLSTM(df)
      response = LL.train_model(train_column_name=model_forecasting_metadata['target'],
                                steps=int(model_forecasting_metadata['steps']))
      metrics_scores = {"r2_score":response['r2_score'],
                        "mean_squared_error":response['mean_squared_error'],
                        "root_mean_squared_error":response['root_mean_squared_error'],}
      model_forecasting_metadata["metrics_scores"] = dumps(metrics_scores)
      model_forecasting_metadata["scaled_test_data"] = dumps([value[0] for value in response['scaled_test_data']])
    
    # save model to pkl format
    model_saved_name = f"{model_forecasting_metadata['model_name']}.pkl"
    joblib.dump(response, model_saved_name)
  
    # save model
    url = f'http://{settings.BACKEND_SERVICE_INTERNAL_IP}:{settings.BACKEND_SERVICE_RUNNING_PORT}/forecasting/saveforecasting/'
    requests.post(url, data=model_forecasting_metadata, files={'file': open(model_saved_name, 'rb')})
    
    # update training record to 'completed'
    url = f'http://{settings.BACKEND_SERVICE_INTERNAL_IP}:{settings.BACKEND_SERVICE_RUNNING_PORT}/forecasting/updateforecastingrecord/'
    json = {'id': training_record['id'], 'status':'completed'}
    record = requests.post(url, json=json)
    os.remove(model_saved_name)
    print("training with record id "+ str(record.json()['id']) + " completed")

# this function will return record in json 
# {'id': 5, 'status': 'accepted'}
async def async_train_forecasting_endpoint(request):
    try:
      model_forecasting_metadata = request.POST.dict()
      file = request.FILES['file']
    except:
      return JsonResponse({'message': "input error"})
    df = pd.read_csv(file)
    
    # create training record in main service db
    url = f'http://{settings.BACKEND_SERVICE_INTERNAL_IP}:{settings.BACKEND_SERVICE_RUNNING_PORT}/forecasting/createforecastingrecord/'
    json = {'status':'accepted'}
    record = requests.post(url, json=json)
    
    training_record = {
      'id' : record.json()['id'],
      'status' : record.json()['status'],
    }  
    asyncio.gather(asyncforecastingtrain(df, training_record, model_forecasting_metadata))
    return JsonResponse(training_record)

async def asynctrain(df, training_record, model_metadata):  

    # update training record to 'in progress'
    url = f'http://{settings.BACKEND_SERVICE_INTERNAL_IP}:{settings.BACKEND_SERVICE_RUNNING_PORT}/modeling/updaterecord/'
    json = {'id': training_record['id'], 'status':'in progress'}
    record = requests.post(url, json=json)
    print("training with record id "+ str(record.json()['id']) + " in progress")

    # train model
    if model_metadata['method'] == 'REGRESSION':
      if model_metadata['algorithm'] == 'LINEAR':
        LR = LumbaLinearRegression(df)
        response = LR.train_model(train_column_name=model_metadata['feature'], target_column_name=model_metadata['target'])
        metrics_scores = {"r2_score":response['r2_score'],
                          "mean_absolute_error":response['mean_absolute_error'],
                          "mean_squared_error":response['mean_squared_error']}
        model_metadata["metrics_scores"] = dumps(metrics_scores)
    if model_metadata['method'] == 'CLASSIFICATION':
      if model_metadata['algorithm'] == 'DECISION_TREE':
        DT = LumbaDecisionTreeClassifier(df)
        response = DT.train_model(train_column_names=model_metadata['feature'].split(','), target_column_name=model_metadata['target'])
        metrics_scores = {"accuracy_score":response['accuracy_score'],
                          "f1_score":response['f1_score'],
                          "precision_score":response['precision_score'],
                          "recall_score":response['recall_score']}
        model_metadata["metrics_scores"] = dumps(metrics_scores)
    if model_metadata['method'] == 'REGRESSION':
      if model_metadata['algorithm'] == 'RANDOM_FOREST':
        RFR = LumbaRandomForestRegressor(df)
        response = RFR.train_model(train_column_names=model_metadata['feature'].split(','), target_column_name=model_metadata['target'])
        metrics_scores = {"r2_score":response['r2_score'],
                          "mean_absolute_error":response['mean_absolute_error'],
                          "mean_squared_error":response['mean_squared_error']}
        model_metadata["metrics_scores"] = dumps(metrics_scores)
    if model_metadata['method'] == 'CLUSTERING':
      if model_metadata['algorithm'] == 'KMEANS':
        KM = LumbaKMeans(df)
        response = KM.train_model(train_column_names=model_metadata['feature'].split(','), k=int(model_metadata['n_cluster']))
        model_metadata["metrics_scores"] = "no_metrics"
        labels_predicted_pkl_name = f"{model_metadata['model_name']}_labels_predicted.pkl"
        joblib.dump(response['labels_predicted'], labels_predicted_pkl_name)

    # save model to pkl format
    model_saved_name = f"{model_metadata['model_name']}.pkl"
    joblib.dump(response['model'], model_saved_name)

    # save model
    url = f'http://{settings.BACKEND_SERVICE_INTERNAL_IP}:{settings.BACKEND_SERVICE_RUNNING_PORT}/modeling/save/'
    if model_metadata['method'] == 'CLUSTERING':
      requests.post(url, 
                    data=model_metadata, 
                    files={'file': open(model_saved_name, 'rb'),
                           'labels_predicted': open(labels_predicted_pkl_name, 'rb')})
    else:  
      requests.post(url, 
                    data=model_metadata, 
                    files={'file': open(model_saved_name, 'rb')})
    
    # update training record to 'completed'
    url = f'http://{settings.BACKEND_SERVICE_INTERNAL_IP}:{settings.BACKEND_SERVICE_RUNNING_PORT}/modeling/updaterecord/'
    json = {'id': training_record['id'], 'status':'completed'}
    record = requests.post(url, json=json)
    os.remove(model_saved_name)
    if model_metadata['algorithm'] == 'KMEANS':
      os.remove(labels_predicted_pkl_name)
    print("training with record id "+ str(record.json()['id']) + " completed")

# this function will return record in json 
# {'id': 5, 'status': 'accepted'}
async def async_train_endpoint(request):
    try:
      model_metadata = request.POST.dict()
      dataset = request.FILES.get('file')
    except:
      return JsonResponse({'message': "input error"})
      # create training record in main service db
    if (model_metadata['type'] != 'object_segmentation'):
      url = f'http://{settings.BACKEND_SERVICE_INTERNAL_IP}:{settings.BACKEND_SERVICE_RUNNING_PORT}/modeling/createrecord/'
      json = {'status':'accepted'}
      record = requests.post(url, json=json)
      
      training_record = {
        'id' : record.json()['id'],
        'status' : record.json()['status'],
      }  
      df = pd.read_csv(dataset)
      
      asyncio.gather(asynctrain(df, training_record, model_metadata))
      return JsonResponse(training_record)
    else:
      asyncio.gather(asyncobjectsegmentationtrain(dataset, model_metadata))
      training_record = {
        'id' : model_metadata['id'],
        'status' : 'accepted',
      }  
      return JsonResponse(training_record)