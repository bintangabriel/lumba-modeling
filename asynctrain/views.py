import asyncio
import json
import os
from django.http import JsonResponse, FileResponse, Http404
import pandas as pd
from ml_model.models.linear_regression import LumbaLinearRegression
from ml_model.models.decision_tree import LumbaDecisionTreeClassifier
from ml_model.models.random_forest import LumbaRandomForestRegressor
from ml_model.models.kmeans import LumbaKMeans
from ml_model.models.arima import LumbaARIMA
import requests
import joblib
from json import dumps
import numpy as np
from modeling import settings
import time
from .obseg_views import *
import io
import zipfile
from modeling.app_redis import Redis

# this function will return record in json 
# {'id': 5, 'status': 'accepted'}
async def async_train_endpoint(request):
  try:
    model_metadata = request.POST.dict()
    r = Redis.get()
    print(model_metadata['file_key'])
    dataset = r.get(model_metadata['file_key'])
    dataset = io.BytesIO(dataset)
    print(dataset)
    print('metadata: ', model_metadata)
    print('dataset: ', dataset)
  except:
    return JsonResponse({'message': "input error"})
  
  asyncio.gather(asyncobjectsegmentationtrain(dataset, model_metadata))
  training_record = {
    'id' : model_metadata['id'],
    'status' : 'accepted',
  }  
  return JsonResponse(training_record)
    
def download_model(req):
  try:
    model_metadata = json.loads(req.body)
    print(model_metadata)
  except Exception as e:
    print(e)
    return JsonResponse({'message': "input error"})
  model_name = model_metadata['model_name']
  username = model_metadata['username']
  workspace = model_metadata['workspace']
  base_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  weights_file = os.path.join(base_directory, 'ml_model', 'models', 'weights', f'{model_name}_{username}_{workspace}.pth')

  if (os.path.exists(weights_file)):
    return FileResponse(open(weights_file, 'rb'), as_attachment=True)
  else:
    Http404("File not found")