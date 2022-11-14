import asyncio
import httpx
from django.http import JsonResponse
import pandas as pd
from ml_model.linear_regression import LumbaLinearRegression
import requests
import joblib

async def asynctrain(df, training_record):  

    # update training record to 'in progress'
    url = 'http://127.0.0.1:8000/modeling/updaterecord/'
    json = {'id': training_record['id'], 'status':'in progress'}
    record = requests.post(url, json=json)
    print("training with record id "+ str(record.json()['id']) + " in progress")

    # for num in range(1,6):
    #   await asyncio.sleep(delay=2)
    #   print("sleep", num)
    
    # train model
    LR = LumbaLinearRegression(df)
    response = LR.train_model(train_column_name='arr_flights', target_column_name='carrier_delay')
    mlmodel = joblib.dump(response['model'], 'mlmodel.pkl')
    
    # update training record to 'completed'
    url = 'http://127.0.0.1:8000/modeling/updaterecord/'
    json = {'id': training_record['id'], 'status':'completed'}
    record = requests.post(url, json=json)
    print("training with record id "+ str(record.json()['id']) + " completed")

# this function will return record in json 
# {'id': 5, 'status': 'accepted'}
async def async_train_endpoint(request):
    try:
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
    asyncio.gather(asynctrain(df, training_record))
    return JsonResponse(training_record)
