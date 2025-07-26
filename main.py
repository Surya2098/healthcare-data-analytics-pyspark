# Streamlit + PySpark Healthcare Dashboard (Complete Version)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, to_date, round as pyspark_round, datediff
)

@st.cache_data
def load_data():
    spark = SparkSession.builder \
        .appName("Healthcare Data Analysis") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()

    df = spark.read.csv("C://Users//SURYA//Downloads//full_medical_data_1000rows.csv", header=True, inferSchema=True)
    df = df.select([col(c).alias(c.replace(" ", "_")) for c in df.columns])
    df = df.dropDuplicates()
    df = df.withColumn("Date_of_Admission", to_date(col('Date_of_Admission'), 'dd-MM-yyyy'))
    df = df.withColumn("Discharge_Date", to_date(col('Discharge_Date'), 'dd-MM-yyyy'))
    df = df.withColumn("Billing_Amount", pyspark_round(col("Billing_Amount"), 2))
    df = df.withColumn("Billing_Amount", when(col("Billing_Amount") < 0, 0).otherwise(col("Billing_Amount")))
    df = df.withColumn("Length_of_Stay", datediff(col("Discharge_Date"), col("Date_of_Admission")))
    return df.toPandas()

# Load and clean data
df = load_data()
df["Medical_Condition"] = df["Medical_Condition"].astype(str).str.strip().str.title()
df["Readmission"] = df["Readmission"].astype(str).str.strip().str.title()
df["Survived"] = df["Survived"].astype(str).str.strip().str.title()

# Streamlit UI
st.set_page_config(page_title="Healthcare Dashboard", layout="wide")
st.title("🏥 Healthcare Data Dashboard")

# Sidebar filters
st.sidebar.header("Filters")
medical_conditions = df["Medical_Condition"].dropna().unique().tolist()
condition_filter = st.sidebar.multiselect("Medical Condition", medical_conditions, default=medical_conditions)

readmissions = df["Readmission"].dropna().unique().tolist()
readmission_filter = st.sidebar.multiselect("Readmission", readmissions, default=readmissions)

survivals = df["Survived"].dropna().unique().tolist()
survival_filter = st.sidebar.multiselect("Survived", survivals, default=survivals)

# Filter data
filtered_df = df[
    df["Medical_Condition"].isin(condition_filter) &
    df["Readmission"].isin(readmission_filter) &
    df["Survived"].isin(survival_filter)
]

# Diagnostics
st.subheader("🧪 Filter Diagnostics")
st.write("Filtered Data Shape:", filtered_df.shape)
if filtered_df.empty:
    st.warning("⚠️ No data found for the selected filters. Please adjust your selections.")
    st.stop()

# Preview
st.subheader("📋 Filtered Data Preview")
st.dataframe(filtered_df.head())

# Summary Stats
st.subheader("📊 Summary Statistics")
st.write(filtered_df.describe())

# Key Metrics
st.subheader("📈 Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("👥 Total Patients", f"{len(filtered_df)}")
col2.metric("💰 Avg Billing Amount", f"₹{filtered_df['Billing_Amount'].mean():.2f}")
col3.metric("🛏️ Avg Length of Stay", f"{filtered_df['Length_of_Stay'].mean():.1f} days")
survival_rate = (filtered_df["Survived"].str.lower() == "yes").mean() * 100
col4.metric("❤️ Survival Rate", f"{survival_rate:.1f}%")

# Distribution plots
st.subheader("📉 Distributions")
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
try:
    sns.histplot(filtered_df["Age"], kde=True, ax=ax[0]); ax[0].set_title("Age")
    sns.histplot(filtered_df["Billing_Amount"], kde=True, ax=ax[1]); ax[1].set_title("Billing Amount")
    sns.histplot(filtered_df["Length_of_Stay"], kde=True, ax=ax[2]); ax[2].set_title("Length of Stay")
    st.pyplot(fig)
except Exception as e:
    st.error(f"Error in distribution plots: {e}")

# Bar chart: Avg Length of Stay by Condition
st.subheader("📊 Average Length of Stay by Condition")
try:
    fig2, ax2 = plt.subplots()
    avg_los = filtered_df.groupby("Medical_Condition")["Length_of_Stay"].mean().reset_index()
    sns.barplot(x="Medical_Condition", y="Length_of_Stay", data=avg_los, ax=ax2)
    plt.xticks(rotation=45)
    st.pyplot(fig2)
except Exception as e:
    st.error(f"Error in bar chart: {e}")

# Pie Charts
st.subheader("🥧 Survival Distribution")
try:
    survival_counts = filtered_df["Survived"].value_counts()
    fig3, ax3 = plt.subplots()
    ax3.pie(survival_counts, labels=survival_counts.index, autopct='%1.1f%%', startangle=90)
    ax3.axis('equal')
    st.pyplot(fig3)
except Exception as e:
    st.error(f"Error in survival pie chart: {e}")

st.subheader("🔄 Readmission Distribution")
try:
    readmit_counts = filtered_df["Readmission"].value_counts()
    fig4, ax4 = plt.subplots()
    ax4.pie(readmit_counts, labels=readmit_counts.index, autopct='%1.1f%%', startangle=90)
    ax4.axis('equal')
    st.pyplot(fig4)
except Exception as e:
    st.error(f"Error in readmission pie chart: {e}")

# Export
st.subheader("📥 Download Filtered Data")
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button("Download as CSV", data=csv, file_name="filtered_healthcare_data.csv", mime="text/csv")
