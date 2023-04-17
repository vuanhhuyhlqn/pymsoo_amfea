FROM python:3.9 
COPY requirements.txt /workspace/requirements.txt
WORKDIR /workspace/ 
RUN pip install -r requirements.txt 

CMD echo 'helloworld'