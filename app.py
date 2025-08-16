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

# Tujuan
st.subheader("Tujuan")
st.markdown("""
Menerapkan regresi teratur (regularized regression) untuk prediksi harga rumah menggunakan **Boston Housing Dataset**.
""")

# --- Boxplot ---
st.subheader("Boxplot : Distribusi Data dan Outlier")
st.markdown("""
Menunjukkan distribusi data, outlier, dan rentang nilai setiap fitur.  
Banyak fitur (`crim`, `zn`, `indus`, `rm`, `age`, `dis`, `rad`, `tax`, `ptratio`, `black`, `lstat`) memiliki pencilan.  
Fitur `tax` dan `black` memiliki rentang nilai luas dengan banyak outlier atas.  
`chas` adalah variabel biner.
""")

fig, ax = plt.subplots(figsize=(15, 8))
sns.boxplot(data=df, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
st.pyplot(fig)

st.markdown("""
**Insight :**  
Beberapa fitur seperti `crim`, `zn`, dan `black` memiliki outlier ekstrem.  
Di tahap EDA, outlier cukup dicatat karena bisa mewakili kondisi nyata dan belum tentu merusak model.
""")

# --- Heatmap ---
st.subheader("Heatmap : Korelasi Antar Variabel")
st.markdown("""
Memvisualisasikan korespondensi antar variabel dengan warna dan angka.  
- **Warna merah :** korelasi positif kuat.  
- **Warna biru :** korelasi negatif kuat.  
- **Angka :** nilai koefisien Pearson (-1 hingga +1).
""")

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", square=True, ax=ax)
st.pyplot(fig)

st.markdown("""
**Insight :**  
Fitur `rm` (jumlah kamar) punya korelasi positif kuat ke `medv`.  
Sebaliknya, `lstat` dan `nox` punya korelasi negatif, menunjukkan bahwa kondisi sosial ekonomi rendah dan polusi udara menurunkan harga rumah.
""")
