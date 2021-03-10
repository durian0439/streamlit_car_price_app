import streamlit as st
import pandas as pd
import numpy as np
# import tensorflow as tf 
# from sklearn.preprocessing import MinMaxScaler

# import matplotlib.pyplot as plt
# import matplotlib
# import seaborn as sb


def main():
    st.title('조건에 따른 자동차 구매 금액 예측하기.')
    df = pd.read_csv('data\Car_Purchasing_Data.csv')
    st.dataframe(df)
    if st.button('최고 연봉자 보기'):
        st.dataframe(df.loc[df['Annual Salary']==df['Annual Salary'].max(),])
    if  st.button('최연소자 확인하기'):
        st.dataframe(df.loc[df['Age']==df['Age'].min(),])

    st.write("딥러닝을 통한 자동차 구매 금액 예측기입니다.")

    gender = st.number_input('성별을 입력해 주세요. 여자는 1, 남자는 0')
    Age = st.number_input('나이를 입력해 주세요')
    Annual_Salary = st.number_input('연봉을 달러화로 입력해 주세요')
    Credit_Card_Debt = st.number_input('카드값을 입력해주세요')
    Net_Worth = st.number_input('총 재산을 달러로 입력해 주세요')

    # new_data = np.array([gender, Age, Annual_Salary, Credit_Card_Debt, Net_Worth])
    # new_data=new_data.reshape(1,-1)
    # st.write(new_data)

    # new_data_scaled = sc.transform(new_data)

    # model = tf.keras.models.load_model("\Car_purchase_model.h5")
    # prediction = prediction_result(model, img)
