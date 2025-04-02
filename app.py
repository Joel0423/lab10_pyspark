import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, mean, when
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import LogisticRegression
import pandas as pd

def create_spark_session():
    return SparkSession.builder.appName("ProductReviewAnalysis").getOrCreate()

@st.cache_resource

def load_data(spark, file):
    return spark.read.csv(file, header=True, inferSchema=True)

def clean_data(df):
    df = df.dropDuplicates()
    df = df.dropna()
    return df

def handle_missing_values(df):
    for col_name in df.columns:
        df = df.fillna({col_name: df.select(mean(col(col_name))).collect()[0][0]})
    return df

def exploratory_data_analysis(df):
    return df.describe().toPandas()

def regression_model(df):
    assembler = VectorAssembler(inputCols=['Age', 'Positive Feedback Count'], outputCol='features')
    df = assembler.transform(df)
    lr = LinearRegression(featuresCol='features', labelCol='Rating')
    model = lr.fit(df)
    return model.summary.r2

def clustering_model(df):
    assembler = VectorAssembler(inputCols=['Age', 'Positive Feedback Count'], outputCol='features')
    df = assembler.transform(df)
    kmeans = KMeans(k=3, seed=1, featuresCol='features')
    model = kmeans.fit(df)
    return model.clusterCenters()

def classification_model(df):
    indexer = StringIndexer(inputCol='Recommended IND', outputCol='label')
    df = indexer.fit(df).transform(df)
    assembler = VectorAssembler(inputCols=['Age', 'Positive Feedback Count'], outputCol='features')
    df = assembler.transform(df)
    lr = LogisticRegression(featuresCol='features', labelCol='label')
    model = lr.fit(df)
    return model.summary.accuracy

st.title("Product Review Analysis using PySpark")

spark = create_spark_session()
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    df_pandas = pd.read_csv(uploaded_file)
    df = spark.createDataFrame(df_pandas)
    st.write("### Raw Data")
    st.write(df.show(5))

    df_cleaned = clean_data(df)
    st.write("### Cleaned Data")
    st.write(df_cleaned.show(5))

    df_no_missing = handle_missing_values(df_cleaned)
    st.write("### Data after Handling Missing Values")
    st.write(df_no_missing.show(5))

    st.write("### Exploratory Data Analysis")
    st.write(exploratory_data_analysis(df_no_missing))

    st.write("### Regression Model Performance (R2 Score)")
    st.write(regression_model(df_no_missing))

    st.write("### Clustering Model (Cluster Centers)")
    st.write(clustering_model(df_no_missing))

    st.write("### Classification Model Accuracy")
    st.write(classification_model(df_no_missing))
