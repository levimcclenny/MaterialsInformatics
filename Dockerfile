FROM tensorflow/tensorflow:1.13.1-gpu-py3

RUN apt-get update && apt-get install -y eog python-tk python-yaml && apt-get clean && rm -rf /var/lib/apt/lists

RUN pip install librosa pytz matplotlib scikit-learn Pillow keras pandas progress

ENV QT_X11_NO_MITSHM=1

RUN mkdir /app

CMD tail -f /dev/null