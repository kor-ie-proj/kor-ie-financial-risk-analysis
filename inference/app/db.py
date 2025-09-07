import os
from sqlalchemy import create_engine
def engine():
    user = os.getenv("DB_USER")
    pwd = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST", "mysql")
    port = os.getenv("DB_PORT", "3306")
    db   = os.getenv("DB_NAME", "riskdb")
    return create_engine(f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}",
                         pool_pre_ping=True, pool_recycle=3600)
