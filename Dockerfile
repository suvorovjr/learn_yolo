FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /app/

RUN sudo apt-get install -y libgl1-mesa-glx

CMD python train.py && tail -f /dev/null
