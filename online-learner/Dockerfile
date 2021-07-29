FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

# Note: should version control these things
RUN pip install tensorflow==2.4.1

RUN pip install vowpalwabbit
RUN pip install python-multipart
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python
RUN pip install pillow

COPY ./model /model/

COPY ./app /app