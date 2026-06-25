# Databricks notebook source
import os

# 1. Procuramos pela pasta dados_csv subindo um nível ou no diretório principal do usuário
caminho_base = os.path.abspath(os.path.join(os.getcwd(), ".."))
caminho_csv = os.path.join(caminho_base, "dados_csv", "application_train.csv")

# Se o caminho acima não achar, o Python tenta o caminho direto do Workspace
if not os.path.exists(caminho_csv):
    # Substitua 'seu_email_aqui' pelo seu e-mail de login do Databricks se necessário
    caminho_csv = "/Workspace/Users/nogsposito@gmail.com/credit-risk/dados_csv/application_train.csv"

print(f"Buscando os dados no caminho absoluto: {caminho_csv}\n")

# 2. O PySpark abre o arquivo de forma distribuída
df_clientes = spark.read.csv(caminho_csv, header=True, inferSchema=True)

# 3. Mostra os dados na tela
display(df_clientes)

# COMMAND ----------

# MAGIC %md
# MAGIC Exploratory Analysis

# COMMAND ----------

from pyspark.sql import functions as F

print("--- TARGET ANALYSIS ---")
# Discovering the proportion of Good (0) vs Bad Payers (1)
df_clientes.groupBy("TARGET").count().withColumn(
    "Percentage (%)", 
    F.round((F.col("count") / df_clientes.count()) * 100, 2)
).show()

print("--- INVESTIGATING COLUMNS WITH NULL VALUES ---")
# Checking for missing data in vital credit information
df_clientes.select(
    F.count(F.when(F.col("AMT_INCOME_TOTAL").isNull(), 1)).alias("Null_Income"),
    F.count(F.when(F.col("AMT_CREDIT").isNull(), 1)).alias("Null_Credit_Amount"),
    F.count(F.when(F.col("OCCUPATION_TYPE").isNull(), 1)).alias("Null_Occupation")
).show()

# COMMAND ----------

# MAGIC %md
# MAGIC Feature Engineering

# COMMAND ----------

from pyspark.sql import functions as F

print("--- STARTING CLEANING AND FEATURE ENGINEERING ---\n")

# Filling nulls with 'Unknown' and creating the new financial feature "CREDIT/INCOME RATIO"
df_treated = df_clientes \
    .fillna({"OCCUPATION_TYPE": "Unknown"}) \
    .withColumn("CREDIT_INCOME_RATIO", F.round(F.col("AMT_CREDIT") / F.col("AMT_INCOME_TOTAL"), 2))

# Checking if nulls disappeared and if the new column was created
print("--- POST-CLEANING CHECK ---")
df_treated.select(
    F.count(F.when(F.col("OCCUPATION_TYPE") == "Unknown", 1)).alias("Unknown_Occupations")
).show()

print("--- NEW FINANCIAL FEATURE SAMPLE ---")
# Displaying a sample for inspection
df_treated.select("AMT_INCOME_TOTAL", "AMT_CREDIT", "CREDIT_INCOME_RATIO", "TARGET").show(5)

# COMMAND ----------

import os
from pyspark.sql import functions as F

base_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
bureau_path = os.path.join(base_path, "dados_csv", "bureau.csv")

if not os.path.exists(bureau_path):
    bureau_path = "/Workspace/Users/nogsposito@gmail.com/credit-risk/dados_csv/bureau.csv"

print('Reading external history data...')
df_bureau = spark.read.csv(bureau_path, header=True, inferSchema=True)

print("--- AGGREGATING EXTERNAL DEBT BEHAVIOR PER CLIENT ---")
df_bureau_agg = df_bureau.groupby("SK_ID_CURR").agg(
    F.count("SK_ID_BUREAU").alias("TOTAL_PREVIOUS_LOANS"),
    F.round(F.sum("AMT_CREDIT_SUM_DEBT"), 2).alias("TOTAL_EXTERNAL_DEBT")
)

df_bureau_agg = df_bureau_agg.fillna({"TOTAL_EXTERNAL_DEBT": 0.0})

df_bureau_agg.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC Combining Data

# COMMAND ----------

from pyspark.sql import functions as F

print("--- STARTING TABLE JOIN (MAIN + BUREAU HISTORY) ---\n")

# Performing the Left Join using the client unique ID
df_final = df_treated.join(df_bureau_agg, on="SK_ID_CURR", how="left")

# Handing the clients who had no external history (filling Nulls with 0)
df_final = df_final.fillna({
    "TOTAL_PREVIOUS_LOANS": 0,
    "TOTAL_EXTERNAL_DEBT": 0.0
})

print("--- FINAL MASTER TABLE SAMPLE ---")
# Displaying key columns to inspect the joined result
df_final.select(
    "SK_ID_CURR", 
    "AMT_INCOME_TOTAL", 
    "CREDIT_INCOME_RATIO", 
    "TOTAL_PREVIOUS_LOANS", 
    "TOTAL_EXTERNAL_DEBT", 
    "TARGET"
).show(5)

print(f"Total rows in Master Table: {df_final.count()}")
