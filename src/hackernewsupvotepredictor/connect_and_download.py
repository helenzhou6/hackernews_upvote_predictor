from psycopg2 import connect, sql
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_USERNAME = os.getenv('POSTGRES_USERNAME')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')

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
def fetch_subs():
    conn =  connect_to_db()
    cur = conn.cursor()
    cur.execute("SELECT i.id, i.by, i.title, i.score, i.url, i.time, u.created as user_created, u.karma as user_karma FROM hacker_news.items i inner join hacker_news.users u on i.by=u.id WHERE i.type='story' AND i.title IS NOT NULL LIMIT 10000;")
    records = cur.fetchall()
    close_db_connection(conn)
    return records

def fetch_users():
    conn =  connect_to_db()
    cur = conn.cursor()
    cur.execute("SELECT id, created FROM hacker_news.users LIMIT 10000;")
    records = cur.fetchall()
    close_db_connection(conn)
    return records 

df_subs = pd.DataFrame(fetch_subs(), columns = ['id','by','title','score','url','time','user_created','user_karma']) 
df_subs.head()
df_subs.info()

# Create a folder called data to be able to run the below
df_subs.to_parquet("data/submissions.parquet.gzip", compression='gzip')  
