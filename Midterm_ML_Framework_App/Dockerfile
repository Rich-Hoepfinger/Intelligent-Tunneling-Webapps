# Set base image (host OS)
FROM python:3.8

# By default, listen on port 5000
EXPOSE 5000/tcp

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY . /app

# Install any dependencies
RUN pip install -r requirements.txt
RUN pip install tensorflow==2.11.0
RUN pip install protobuf==3.20.*

# Specify the command to run on container start
ENV FLASK_APP=app.py
CMD [ "flask", "run", "--host=0.0.0.0" ]