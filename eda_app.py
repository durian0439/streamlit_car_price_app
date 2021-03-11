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
import string as str


def run_eda_app() :
    st.subheader('EDA 화면입니다.')

    car_df = pd.read_csv('data/Car_Purchasing_Data.csv',encoding='ISO-8859-1')

    radio_menu = ['데이터프레임', '통계치']
    selected_radio = st.radio('선택하세요',radio_menu)

    if selected_radio == '데이터프레임' :
        st.dataframe(car_df)
    elif selected_radio == '통계치' :
        st.dataframe(car_df.describe())

    columns = car_df.columns
    columns = list(columns)

    multi=st.multiselect('컬럼을 선택해주세요', columns)
    if len(multi) != 0 :
        st.dataframe(car_df[multi])
    else:
        st.write('컬럼이 없습니다.')
    # st.dataframe(car_df[multi])

    #상관계수를 확인.
    #멀티셀렉트에 선택
    #해당컬럼에대한 상관계수
    #단,숫자데이터에 대한 상관관계

    # corr_df = car_df.iloc[:,3:].columns   #이렇게 하면 컬럼 변경 시 사용불가
    # # print(corr_df)

    # multi_corr=st.multiselect('상관관계 확인을 위해 컬럼을 선택해주세요', corr_df)
    # st.dataframe(car_df[multi_corr].corr())


    corr_columns = car_df.describe().columns.values # describe를 사용하면 숫자데이터만 나타내기에 사용.
    multi_corr_columns=st.multiselect('상관관계 확인을 위해 컬럼을 선택해주세요', corr_columns)
    if len(multi_corr_columns) != 0:
        st.dataframe(car_df[multi_corr_columns].corr())

        fig = plt.figure()  # 스트림릿에서 plot그리기
        plt.title('corr plot')
        st.pyplot(sns.pairplot(data = car_df[multi_corr_columns]))
    else:
        st.write('컬럼이 없습니다.')


    # 컬럼을 하나만 선택하면, 해당컬럼의 min/max에 해당하는 사람의 데이터를 화면에 표출

    corr_columns = car_df.describe().columns.values # describe를 사용하면 숫자데이터만 나타내기에 사용.
    multi_corr_columns=st.selectbox('최대 최소값을 보기위해 컬럼을 선택해주세요', corr_columns)
    if len(multi_corr_columns) != 0:
        st.dataframe(car_df[car_df[multi_corr_columns]==car_df[multi_corr_columns].min()])
        st.dataframe(car_df[car_df[multi_corr_columns]==car_df[multi_corr_columns].max()])
    else:
        st.write('컬럼이 없습니다.')

    # 고객 이름검색 툴 만들기
    
    word=st.text_input('검색어를 입력하세요')
    word = word.lower()
    result = car_df.loc[car_df['Customer Name'].str.contains(word, case = False, )] #대소문자 상관없이 찾으라

    st.dataframe(result)
