from django.shortcuts import render
from rest_framework.decorators import api_view
import asyncio
from time import sleep
import httpx
from django.http import HttpResponse

# Create your views here.

@api_view()
def asynctrain(request):
    pass

async def http_call_async():
  for num in range(1,6):
    await asyncio.sleep(delay=2)
    print("test", num)
  async with httpx.AsyncClient() as client:
    r = await client.get("https://httpbin.org")
    print(r)

async def async_view(request):
  asyncio.gather(asyncio.create_task(http_call_async()))
  return HttpResponse('Non-blocking HTTP request, succeed')