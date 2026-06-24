# Databricks notebook source
# MAGIC %pip install kaggle

# COMMAND ----------

import os

os.environ['KAGGLE_API_TOKEN'] = "KGAT_b182e1a3250d12674130482f8fedc9a9"

# COMMAND ----------

# MAGIC %sh
# MAGIC kaggle competitions download -c home-credit-default-risk

# COMMAND ----------

dbutils.fs.mkdirs("dbfs:/FileStore/home_credit_data")

# COMMAND ----------

import zipfile
import os

caminho_zip = "home-credit-default-risk.zip"
pasta_destino = "dados_csv"

print("Iniciando a extração dos 2GB de dados... Isso pode levar um minutinho.")

os.makedirs(pasta_destino, exist_ok=True)

# Extrair tudo
with zipfile.ZipFile(caminho_zip, 'r') as zip_ref:
    zip_ref.extractall(pasta_destino)
    
print(f"Extração concluída com sucesso! Verifique a pasta '{pasta_destino}' na barra lateral.")

# COMMAND ----------

import os

# 1. O Python descobre o caminho absoluto REAL que o Databricks Serverless exige
caminho_absoluto = os.path.abspath("dados_csv/application_train.csv")

print(f"O caminho que o Spark vai ler é: {caminho_absoluto}\n")

# 2. O Spark agora lê o arquivo sabendo exatamente que ele está no Workspace
df_clientes = spark.read.csv(caminho_absoluto, header=True, inferSchema=True)

# 3. Mostra a tabela na tela
display(df_clientes)

# 4. Conta o total de registros
print(f"\nTotal de clientes carregados com sucesso: {df_clientes.count()}")
