import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import h5py
import tensorflow.keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
import pickle
import streamlit as st
import joblib

def run_ml_app():
    st.subheader('Machine Learning')

    #1. 유저에게 정보 입력을 받음
    # 성별
    gender = st.radio('성별을 입력', ['남자','여자'])
    if gender == '남자':
        gender = 1
    else:
        gender = 0
    
    age = st.number_input('나이입력',min_value=0, max_value=120)
    salary = st.number_input('연봉입력', min_value=0)
    debt = st.number_input('빚 입력', min_value=0)
    worth = st.number_input('자산 입력',min_value=0)

    # 2. 예측한다.
    # 2-1. 모델 불러오기
    model = tensorflow.keras.models.load_model('data/Car_purchase_model.h5')

    # 2-2. numpy array 만들기
    new_data = np.array([gender, age, salary, debt, worth]) 

    # 2-3. 피쳐스케일링 하기
    new_data = new_data.reshape(1,-1)
    sc_X = joblib.load('data/sc_X.pkl')

    new_data_scaled = sc_X.transform(new_data)

    # 2-4. 예측한다.
    y_pred = model.predict(new_data_scaled) 

    # 에측 결과는 스케일링 된 결과이므로, 다시 돌려야한다.
    # st.write(y_pred[0][0])
    sc_y = joblib.load('data/sc_y.pkl')
    y_pred_original = sc_y.inverse_transform(y_pred)

    # 3. 결과를 화면에 보여준다.
    btn = st.button('결과보기')

    st.write('예측 결과입니다. {:,.1f}$ 의 차를 살 수 있습니다.'.format(y_pred_original[0,0]))
    st.write(y_pred_original)