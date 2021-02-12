FROM borda/docker_python-opencv-ffmpeg

WORKDIR /app

COPY requirements.txt /app
RUN pip install --trusted-host pypi.python.org -r requirements.txt

Add . /app

CMD ["python", "etl.py"]
