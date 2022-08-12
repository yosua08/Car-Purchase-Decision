from multiprocessing import dummy
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle

st.write("""
# Klasifikasi Pembeli mobil (Web App)
Aplikasi berbasis web untuk memprediksi pembeli mobil  
Dataset didapat dari kaggle https://www.kaggle.com/datasets/gabrielsantello/cars-purchase-decision-dataset
""")

st.sidebar.header('Parameter Inputan')

# Upload File CSV untuk parameter inputan
#upload_file = st.sidebar.file_uploader('Upload file CSV Anda', type=['csv'])
#if upload_file is not None:
#    inputan = pd.read_csv(upload_file)
    
#else:
def input_user():
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    umur = st.sidebar.slider('Age', value=20, min_value=17, max_value=80, step=1)
    gaji = st.sidebar.slider('AnnualSalary', value=20000, min_value=15000, max_value=155000, step=1000)
    data = {'Gender' : gender,
            'Age' : umur,
            'AnnualSalary' : gaji}
    fitur = pd.DataFrame(data, index=[0])
    return fitur
inputan = input_user()

# Menggabungkan inputan dan dataset car    
car_raw = pd.read_csv('car_data.csv')
cars = car_raw.drop(columns=['Purchased', 'User ID'])
    
df = pd.concat([inputan, cars], axis=0)

# Encode 
encode = ['Gender']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
df = df[:1]

# Menampilkan parameter
st.subheader('Parameter Inputan')

#if upload_file is not None:
#    st.write(df)
#else:
st.write('Menunggu file csv untuk diupload. Saat ini menggunakan sample inputan')
st.write(df)
    
# Load model NBC
load_model = pickle.load(open('NBC_car.pkl', 'rb'))

# Terapkan NBC
prediksi = load_model.predict(df)
prediksi_proba = load_model.predict_proba(df)

st.subheader('Keterangan Label Kelas')
purcase = np.array(['Membeli', 'Tidak Membeli'])
st.write(purcase)

st.subheader('Hasil Prediksi')
st.write(purcase[prediksi])