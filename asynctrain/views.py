import asyncio
import httpx
from django.http import JsonResponse
import pandas as pd
from ml_model.linear_regression import LumbaLinearRegression

async def asynctrain(request):
    try:
      file = request.FILES['file']
    except:
      return JsonResponse({'message': "input error"})
    
    df = pd.read_csv(file)
    LR = LumbaLinearRegression(df)
    response = LR.train_model(train_column_name='arr_flights', target_column_name='carrier_delay')
    return JsonResponse({'message': 'model created', 'mean_absolute_error':response['mean_absolute_error']})

async def http_call_async():
  for num in range(1,6):
    await asyncio.sleep(delay=2)
    print("test", num)
  async with httpx.AsyncClient() as client:
    r = await client.get("https://httpbin.org", timeout=None)
    print(r)

async def async_view(request):
  if request.method == 'POST':
    print(request)
  asyncio.gather(asyncio.create_task(http_call_async()))
  return JsonResponse({'message':'Non-blocking HTTP request, succeed'})