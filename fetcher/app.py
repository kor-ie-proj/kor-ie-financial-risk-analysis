import os, time, datetime
from sqlalchemy import create_engine, text

def engine():
    user = os.getenv("DB_USER")
    pwd = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST", "mysql")
    port = os.getenv("DB_PORT", "3306")
    db   = os.getenv("DB_NAME", "riskdb")
    return create_engine(f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}", pool_pre_ping=True)

eng = engine()
interval = int(os.getenv("INTERVAL_SEC", "60"))

while True:
    now = datetime.date.today()
    with eng.begin() as conn:
        conn.execute(text("""
          INSERT INTO economic_indicators (indicator_date, base_rate, ccsi)
          VALUES (:d, 3.25, 99)
        """), {"d": now})
    print(f"[fetcher] inserted mock indicators for {now}")
    time.sleep(interval)
