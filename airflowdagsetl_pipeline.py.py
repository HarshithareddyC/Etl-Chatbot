# airflow/dags/etl_pipeline.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import json
from etl_engine import execute_etl_step

def run_etl():
    df = pd.read_csv("uploaded_data.csv")  # Pre-saved copy
    with open("etl_instruction.txt") as f:
        step = json.load(f)
    df = execute_etl_step(df, step)
    df.to_csv("processed_output.txt", sep="\t", index=False)

default_args = {
    'start_date': datetime(2024, 1, 1),
}

with DAG("etl_pipeline", default_args=default_args, schedule_interval=None, catchup=False) as dag:
    etl_task = PythonOperator(
        task_id="run_etl_task",
        python_callable=run_etl
    )
