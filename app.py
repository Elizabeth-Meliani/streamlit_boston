import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("boston.csv") 

# Judul dan deskripsi
st.title("Analisis Faktor Penentu Harga Rumah di Boston")
st.markdown("Aplikasi ini menampilkan visualisasi interaktif dan insight dari dataset Boston Housing.")

# Sidebar profil
st.sidebar.header("Profil")
st.sidebar.markdown("**Nama :** Elizabeth Meliani")
st.sidebar.markdown("**Email :** melzyunho@gmail.com")
st.sidebar.markdown("**Bio :** Data Scientist Learner")

# Pilihan visualisasi
chart_type = st.selectbox("Pilih Visualisasi", ["Boxplot Semua Fitur", "Heatmap Korelasi Antar Variabel"])

if chart_type == "Boxplot Semua Fitur":
    st.subheader("Distribusi dan Outlier pada Semua Fitur")
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.boxplot(data=df, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

    st.markdown("""
    **Insight:**
    Beberapa fitur seperti `crim`, `zn`, dan `black` memiliki outlier ekstrem.
    Di tahap EDA, outlier cukup dicatat karena bisa mewakili kondisi nyata dan belum tentu merusak model.
    """)

elif chart_type == "Heatmap Korelasi Antar Variabel":
    st.subheader("Korelasi Antar Variabel terhadap Harga Rumah (`medv`)")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", square=True, ax=ax)
    st.pyplot(fig)

    st.markdown("""
    **Insight:**
    Fitur `rm` (jumlah kamar) punya korelasi positif kuat ke `medv`.
    Sebaliknya, `lstat` dan `nox` punya korelasi negatif, menunjukkan bahwa kondisi sosial ekonomi rendah dan polusi udara menurunkan harga rumah.
    """)
