import asyncio
import json
from django.http import JsonResponse
import pandas as pd
from ml_model.linear_regression import LumbaLinearRegression
import requests
import joblib

async def asynctrain(df, training_record, model_metadata):  

    # update training record to 'in progress'
    url = 'http://127.0.0.1:8000/modeling/updaterecord/'
    json = {'id': training_record['id'], 'status':'in progress'}
    record = requests.post(url, json=json)
    print("training with record id "+ str(record.json()['id']) + " in progress")

    # train model
    LR = LumbaLinearRegression(df)
    response = LR.train_model(train_column_name='arr_flights', target_column_name='carrier_delay')
    
    # save model to pkl format
    model_saved_name = f"{model_metadata['model_name']}.pkl"
    joblib.dump(response['model'], model_saved_name)

    # save model
    url = 'http://127.0.0.1:8000/modeling/save/'
    requests.post(url, data=model_metadata, files={'file': open(model_saved_name, 'rb')})
    
    # update training record to 'completed'
    url = 'http://127.0.0.1:8000/modeling/updaterecord/'
    json = {'id': training_record['id'], 'status':'completed'}
    record = requests.post(url, json=json)
    print("training with record id "+ str(record.json()['id']) + " completed")

# this function will return record in json 
# {'id': 5, 'status': 'accepted'}
async def async_train_endpoint(request):
    try:
      model_metadata = request.POST.dict()
      file = request.FILES['file']
    except:
      return JsonResponse({'message': "input error"})
    df = pd.read_csv(file)

    # create training record in main service db
    url = 'http://127.0.0.1:8000/modeling/createrecord/'
    json = {'status':'accepted'}
    record = requests.post(url, json=json)
    
    training_record = {
      'id' : record.json()['id'],
      'status' : record.json()['status'],
    }  
    asyncio.gather(asynctrain(df, training_record, model_metadata))
    return JsonResponse(training_record)
