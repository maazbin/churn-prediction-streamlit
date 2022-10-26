# from turtle import width
import pandas as pd
import numpy as np
import streamlit as st
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split


import seaborn as sn
import matplotlib.pyplot as plt

# import numpy as np




# loading data
@st.cache
def load_data():    
    
    return pd.read_csv('/home/maaz/highplains_ml/churn_pred/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    


@st.cache
def preprocess(df):
    
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)

    df_train.TotalCharges = df_train['TotalCharges'].replace(' ', '0', regex=True)
    df_train.TotalCharges = pd.to_numeric(df_train.TotalCharges)

    df_test.TotalCharges = df_test['TotalCharges'].replace(' ', '0', regex=True)
    df_test.TotalCharges = pd.to_numeric(df_test.TotalCharges)

    df_train['Churn'] = df_train['Churn'].map({'Yes':1, 'No':0})
    df_test['Churn'] = df_test['Churn'].map({'Yes':1, 'No':0})

    return df_train,df_test

# @st.cache
def demographic(df_train):
    demo_features = ['gender','SeniorCitizen','Partner','Dependents']

    for feature in demo_features:
        cross_table = df_train[[feature,'Churn']].groupby([feature], as_index=False).mean().sort_values(by='Churn', ascending=False)
        st.write(cross_table)
        st.bar_chart(cross_table,x = feature,width = 400,height  = 0,use_container_width=False)
    
    
    # for feature in demo_features:
    #     st.bar_chart(df_train.Churn[feature],width = 400,height  = 0,use_container_width=False)

# @st.cache
def service_features(df_train):
    service_features = ['PhoneService','MultipleLines','InternetService','OnlineSecurity',
                    'OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']



    for feature in service_features:
        cross_table = df_train[[feature,'Churn']].groupby([feature], as_index=False).mean().sort_values(by='Churn', ascending=False)
        # st.write(cross_table)
        st.bar_chart(cross_table,x = feature,width = 400,height  = 0,use_container_width=False)


# @st.cache
def acc_features(df_train):
    acc_features = ['tenure','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges']
    cat_acc_features = ['Contract','PaperlessBilling','PaymentMethod']
    num_acc_features = ['tenure','MonthlyCharges','TotalCharges']





    for feature in cat_acc_features:
        cross_table = df_train[[feature,'Churn']].groupby([feature], as_index=False).mean().sort_values(by='Churn', ascending=False)
        # st.write(cross_table)
        st.bar_chart(cross_table,x = feature,width = 400,height  = 0,use_container_width=False)

    df_churn = df_train.groupby(['Churn'], as_index=False).mean()

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,4))
    for i in range(3):
        feature = num_acc_features[i]
        sn.barplot(x='Churn',y=feature, data=df_churn, ax=axes[i])
        axes[i].set_title(f"Average {feature}")
        axes[i].set_xlabel(f'Churn')
        axes[i].set_ylabel(f"{feature}")

    # for feature in ['tenure','MonthlyCharges','TotalCharges']:
    #     g = sn.FacetGrid(df_train, col='Churn', height=4)
    #     g.map(plt.hist, feature, bins=50)
    st.pyplot(fig)

    # for feature in ['tenure','MonthlyCharges','TotalCharges']:
    #     st.bar_chart(df_train[feature],width = 400,height  = 0,use_container_width=False)


def churn_rate(df_train):

    st.write(f"Global Churn Rate : {(df_train['Churn'].sum()/len(df_train))*100}%")
    churn = {} 
    churn['No'], churn['Yes'] = df_train.Churn.value_counts()
    a = pd.Series(data = churn,index=churn.keys())

    st.write(a)

    st.bar_chart(a,width = 400,height  = 0,use_container_width=False)

df = load_data()
df_train, df_test = preprocess(df)

st.markdown(

    """
    
# Exploratory Analysis

    1) Churn Rate
    2) Demographic Information: gender,SeniorCitizen,Partner,Dependents
    3) Service Information: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
    4) Account Information: tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges


    """
)



# st.write(df_train_full)

# churn_rate = (df_train['Churn'].value_counts()["Yes"]/df_train['Churn'].count())*100




if st.checkbox('Churn Rate'):
    st.subheader("CHURN RATE")
    churn_rate(df_train)

    # st.write("# CHURN RATE")


#  if user checks Raw data
if st.checkbox('Demographic Information'):
    st.subheader("Demographic Information")
    demographic(df_train)


if st.checkbox("Service Information"):
    st.subheader("### Service Information")

    service_features(df_train)


if st.checkbox("Account Information"):
    st.subheader("Account Information")
    acc_features(df_train)



