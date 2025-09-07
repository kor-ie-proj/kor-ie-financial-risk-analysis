FROM python:3.11-slim
WORKDIR /app
COPY app.py /app/app.py
RUN pip install --no-cache-dir pymysql SQLAlchemy
CMD ["python", "app.py"]
