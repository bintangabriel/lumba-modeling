enable concurrency in a server
run django app as an ASGI

in development: 
`pip install uvicorn`
`uvicorn --reload modeling.asgi:application`

in production:
`pip install uvicorn`
`gunicorn myproject.asgi:application -k uvicorn.workers.UvicornWorker`