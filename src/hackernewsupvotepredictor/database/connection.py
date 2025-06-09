from psycopg2 import connect, sql
from dotenv import load_dotenv
from os import getenv

load_dotenv()

POSTGRES_PASSWORD = getenv('POSTGRES_PASSWORD')
POSTGRES_USERNAME = getenv('POSTGRES_USERNAME')
DB_HOST = getenv('DB_HOST')
DB_PORT = getenv('DB_PORT')
DB_NAME = getenv('DB_NAME')

def connect_to_db():
    conn = connect(
        user=POSTGRES_USERNAME, password=POSTGRES_PASSWORD, host=DB_HOST, port=DB_PORT, database=DB_NAME
    )
    return conn

def close_db_connection(conn):
    conn.commit()
    conn.close()

# Data has COLUMNS: id, dead, type, by, time, text, parent, kids, url, score, title, descandants 

# 1. EXTRACT id, by, title, score, url -- FILTER where title exists
def filter_records():
    conn =  connect_to_db()
    cur = conn.cursor()
    cur.execute("SELECT id, by, title, score, url FROM hacker_news.items WHERE type='story' AND title IS NOT NULL LIMIT 3;")
    records = cur.fetchall()
    close_db_connection(conn)
    return records


print(filter_records())


