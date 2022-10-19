import streamlit as st

import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt
# %matplotlib inline

from sklearn.feature_extraction import DictVectorizer

import joblib

import json
# setting up multi page app

st.set_page_config(

    page_title="Churn Prediction",
    # page_icon="👋",

)

st.write("# Churn Prediction")


# these are yes no  labels for catagorical vars
labels_one_hot = [
    
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'paperlessbilling',
]



# loading data
@st.cache
def load_data():    
    
    return pd.read_csv('../WA_Fn-UseC_-Telco-Customer-Churn.csv')


def selectLabel():

    st.write("### This is yes/no question")
    options = st.multiselect(
        "Please select if a custumer falls in these catagories \n eg: Custumer was given phoneservice",
        labels_one_hot,
        'phoneservice',
    )
    st.write(f'#### You selected: {options}' )
    
    listA = [i for i in labels_one_hot if i not in options]
    
    labels = dict.fromkeys(options, 'yes')
    
    st.write(f'#### You selected for yes: {labels}' )
    
    labels = dict.fromkeys(listA, 'no')
    
    st.write(f'#### You selected for no: {labels}' )


    labels['seniorcitizen'] = senior_citizen = st.checkbox(
        'Senior Citizon',
        (0)
        
        )
    st.write(int(senior_citizen))



    labels['gender'] = gender = st.radio(
        "Please select a gender",
        ('male', 'female')
    )

    # if gender  == 'Male':
    #     labels['gender']  = 1
    # else :
    #     labels['gender']  = 0
    

    st.write(gender)

    labels['internetservice'] = internet_service = st.radio(
        "Internet service type",
        ('dsl', 'fiber_optic', 'no')
    )

    labels['contract'] = contract = st.radio(

        "Please select contract type",
        ['month-to-month', 'one_year', 'two_year']
    )

    labels['paymentmethod'] = payment_method = st.radio(

        "Please select a payment method",
        ['electronic_check', 'mailed_check', 'bank_transfer_(automatic)','credit_card_(automatic)']
    )

    # st.write('###### ')
    labels['tenure'] = tenure = st.number_input('Tenure')
    st.write('Current value : ', tenure)

    labels['monthlycharges'] = monthly_charges = st.number_input('Monthly Charges')
    st.write('Current value : ', monthly_charges)

    labels['totalcharges'] = total_charges = st.number_input('Total Charges')
    st.write('Current value : ', total_charges)
    
    return labels


@st.cache
def load_feature_dict():
    
    with open('../feature_dict.json', 'r') as fp:
        train_dict = json.loads(fp.read())
        dv = DictVectorizer(sparse=False)
        dv.fit(train_dict)
    return dv

# converts dictionary to vector
@st.cache
def dict_vectorizer(labels):
    #loading train dictionary for feature extraction
    dv = load_feature_dict()
    
    transformed_labels = dv.transform([labels])
    return transformed_labels

#loads ml model
@st.cache
def load_model():
    # filename = '../finalized_model_joblib.sav'
    return joblib.load('../finalized_model_joblib.sav')

# predicts churning probability
@st.cache
def churnPrediction(labels,model):
    prediction = model.predict_proba(labels)[0,1]
    return prediction



#  if user checks Raw data
if st.checkbox('Raw Data'):
    st.subheader('Raw data')
    st.write(load_data())



#select labels
labels = selectLabel()

# vectorizing labels 
labels = dict_vectorizer(labels)

# loading ml model
model = load_model()

st.write(f'Churning probability : {churnPrediction(labels,model)}')    

# options = st.multiselect(
#     'What are your favorite colors',
#     ['Green', 'Yellow', 'Red', 'Blue'],
#     ['Yellow', 'Red'])

# st.write('You selected:', options)


# """
#     {'gender': 'male',
#  'seniorcitizen': 0,
#  'partner': 'yes',
#  'dependents': 'no',
#  'phoneservice': 'yes',
#  'multiplelines': 'no',
#  'internetservice': 'dsl',
#  'onlinesecurity': 'yes',
#  'onlinebackup': 'yes',
#  'deviceprotection': 'yes',
#  'techsupport': 'yes',
#  'streamingtv': 'yes',
#  'streamingmovies': 'yes',
#  'contract': 'two_year',
#  'paperlessbilling': 'yes',
#  'paymentmethod': 'bank_transfer_(automatic)',
#  'tenure': 71,
#  'monthlycharges': 86.1,
#  'totalcharges': 6045.9}
# """
# st.write(f"### Labels : {labels} ")




# st.write(df)