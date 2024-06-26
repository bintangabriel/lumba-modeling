# Use an official Python runtime as a parent image
FROM nvcr.io/nvidia/pytorch:24.05-py3

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port uvicorn will run on
EXPOSE 7000

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Run the command to start uvicorn
CMD ["uvicorn", "--reload", "modeling.asgi:application", "--host", "0.0.0.0", "--port", "7000"]
