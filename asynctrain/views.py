import asyncio
import httpx
from django.http import JsonResponse
import pandas as pd
from ml_model.linear_regression import LumbaLinearRegression
import requests

async def asynctrain(df):  
  # LR = LumbaLinearRegression(df)
  # response = LR.train_model(train_column_name='arr_flights', target_column_name='carrier_delay')
  # return JsonResponse({'message': 'model created', 'mean_absolute_error':response['mean_absolute_error']})
  for num in range(1,6):
    await asyncio.sleep(delay=2)
    print("sleep", num)
  print("post request to main service")

# this function will return record in json 
# {'id': 5, 'status': 'accepted'}
async def async_train_endpoint(request):
  try:
      file = request.FILES['file']
  except:
    return JsonResponse({'message': "input error"})
  df = pd.read_csv(file)

  # create record model training in main service db
  url = 'http://127.0.0.1:7000/records/create/'
  json = {'status':'accepted'}
  record = requests.post(url, json=json)
  
  training_record = {
    'training_id' : record.json()['id'],
    'status' : record.json()['status']
  }  
  asyncio.gather(asynctrain(df))
  return JsonResponse(training_record)
