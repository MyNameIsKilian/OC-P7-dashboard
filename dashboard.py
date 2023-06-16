import streamlit as st
import pandas as pd
import numpy as np
import time

st.title('Prêt à dépenser : Scoring Crédit')
st.image('./images/logo-app.png')

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
         'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

h1 = st.subheader('Load data')
with st.spinner(text='In progress'):
  time.sleep(3)
  st.success('Done')
h2 = st.subheader("Select user's id to predict (selectbox or input)")
st.selectbox('', [1,2,3])
st.button('Find user')

# Permettre de visualiser des informations descriptives relatives à un client (via un système de filtre)
h3 = st.subheader("Load user and display it, add filter")
df = pd.DataFrame([range(20)], columns=[f"col {i}" for i in range(20)])
st.dataframe(df)

h4 = st.subheader("Run prediction")
st.button('Predict')

# Permettre de visualiser le score et l’interprétation de ce score pour chaque client de façon intelligible pour une personne non experte en data science
h5 = st.subheader("Display results")
st.text("Granted or Refused")
st.text("Score")

st.subheader("Display SHAP values")
st.line_chart(df.iloc[0,:])

# Permettre de comparer les informations descriptives relatives à un client à l’ensemble des clients ou à un groupe de clients similaires
h7 = st.subheader("Display similar users")
data = np.random.randint(1, 101, size=(10, 20))
df2 = pd.DataFrame(data, columns=[f"col {i}" for i in range(20)])