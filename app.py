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
from eda_app import run_eda_app
from ml_app import run_ml_app

def main():
    st.title('자동차 가격 예측')
    menu = ['Home', 'EDA', 'ML']
    choice = st.sidebar.selectbox("Menu",menu)
    print(choice)

    if choice == 'Home':
        uploaded_files = st.write('이 앱은 고객 데이터와 구매 데이터에 대한 내용입니다. 해당 고객의 정보를 입력하면, 얼마정도의 차를 구매할 수 있는지 예측하는 앱입니다.')
        st.write('왼쪽의 사이드 바에서 선택하세요.')

    elif choice == 'EDA':
        run_eda_app()

    elif choice == 'ML':
        run_ml_app()

if __name__ == '__main__':
    main()
