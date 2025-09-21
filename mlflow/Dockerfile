FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir mlflow==2.14.2 pymysql boto3 sqlalchemy
COPY start.sh /start.sh
RUN chmod +x /start.sh
EXPOSE 5000
CMD ["/start.sh"]