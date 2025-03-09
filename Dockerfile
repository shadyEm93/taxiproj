

#using an official Python runtime as a parent image
FROM python:3.9-slim

#setting the working directory in the container
WORKDIR /app

#copying the requirements file into the container
COPY requirements.txt .

#installing the dependencies
RUN pip install --no-cache-dir -r requirements.txt

#copying the rest of the application code into the container
COPY . .

#exposing any required ports (if your app uses Flask or FastAPI)
EXPOSE 8080

#command to run the application
CMD ["python", "taxifinal.py"]
