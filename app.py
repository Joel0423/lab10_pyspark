import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, mean
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_spark_session():
    return SparkSession.builder.appName("WomensClothingReview").getOrCreate()

# Initialize Streamlit App
st.title("Women's Clothing Reviews - EDA & ML with PySpark")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    spark = create_spark_session()
    df_pandas = pd.read_csv(uploaded_file)
    df = spark.createDataFrame(df_pandas)
    st.write("### Data Sample:")
    st.table(df.limit(50).toPandas())

    # Data Cleaning & Wrangling
    if st.button("Clean Data"):
        df = df.na.drop()
        st.write("### Cleaned Data Sample:")
        st.table(df.limit(50).toPandas())

    if st.button("Perform EDA"):
        pdf = df.toPandas()
        fig, ax = plt.subplots()
        sns.histplot(pdf['Rating'], bins=5, kde=True, ax=ax)
        st.pyplot(fig)

    # Regression
    if st.button("Run Regression"):
        assembler = VectorAssembler(inputCols=['Age', 'Positive Feedback Count'], outputCol='features')
        df = assembler.transform(df).select('features', col('Rating').alias('label'))
        lr = LinearRegression()
        model = lr.fit(df)
        st.write("Regression Model Coefficients:", model.coefficients)
        st.write("Intercept:", model.intercept)
    

    # Clustering
    if st.button("Run Clustering"):
        assembler = VectorAssembler(inputCols=['Age', 'Positive Feedback Count'], outputCol='features')
        df = assembler.transform(df).select('features')
        kmeans = KMeans(k=3)
        model = kmeans.fit(df)
        st.write("Cluster Centers:", model.clusterCenters())

    # Classification
    if st.button("Run Classification"):
        assembler = VectorAssembler(inputCols=['Age', 'Positive Feedback Count'], outputCol='features')
        df = assembler.transform(df).select('features', col('Recommended IND').alias('label'))
        log_reg = LogisticRegression()
        model = log_reg.fit(df)
        st.write("Classification Model Coefficients:", model.coefficients)
        st.write("Intercept:", model.intercept)
