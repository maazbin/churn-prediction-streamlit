import streamlit as st
import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt



from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression




st.write("# Model Evaluation")

st.subheader("Model Name")
st.write("Logistic Regression")

st.subheader("Model Type")
st.write("Classification")

st.subheader("Model Output")
st.write("Binary")



st.subheader("Accuracy")
st.write(0.8016129032258065)
