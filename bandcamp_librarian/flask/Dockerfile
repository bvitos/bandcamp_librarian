FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt /app
RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN python -m spacy download en_core_web_md

Add . /app

CMD ["python", "application.py"]
