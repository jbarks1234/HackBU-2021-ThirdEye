FROM python:3.7-slim

WORKDIR thirdeye

EXPOSE 5000

RUN pip install flask

COPY . .

CMD ["flask", "run", "--host=0.0.0.0"]