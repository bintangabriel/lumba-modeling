run django app as an ASGI to enable asynchronous in a server

1. create virtual environment  
`virtualenv venv`  
2. activate virtual environment  
`.\venv\scripts\activate`  
3. install requirements    
`pip install -r requirements.txt`  
4. run server  
`uvicorn --reload modeling.asgi:application --port 7000`
