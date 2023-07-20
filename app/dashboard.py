import streamlit as st
import pandas as pd
import numpy as np
import requests
import shap
import json
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# Timeout for requests (connect, read)
TIMEOUT = (5, 30)
MAIN_COLUMNS = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
                'NAME_FAMILY_STATUS_Married', 'NAME_INCOME_TYPE_Working',
                'AMT_INCOME_TOTAL', 'PAYMENT_RATE',
                'DAYS_BIRTH', 'DAYS_EMPLOYED']
API_URL = "https://kiliandatadev.pythonanywhere.com/"

def st_shap(plot, height=None):
    """ Create a shap html component """
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# @st.cache
# def get_columns_mean():
#     """ Get customers main columns mean values """
#     response = requests.get(API_URL + "columns/mean", timeout=TIMEOUT)
#     content = json.loads(json.loads(response.content))
#     return pd.Series(content)
 
# @st.cache
# def get_columns_neighbors(cust_id):
#     """ Get customers neighbors main columns mean values """
#     response = requests.get(API_URL + "columns/neighbors/id=" + str(cust_id), timeout=TIMEOUT)
#     content = json.loads(json.loads(response.content))
#     return pd.Series(content)

st.title('Prêt à dépenser : Scoring Crédit')
st.image('./assets/images/logo-app.png', width=250)

st.subheader('Tableau de données de nos clients')
data = pd.read_csv('./data/X_train_sample.csv', index_col=[0])
st.dataframe(data)

st.subheader("Sélectionner un client")
ids_index = data.index

id_selected = st.selectbox('Sélectionner un ID', options=ids_index, index=0)
colonnes_selectionnees = st.multiselect('Filtrer par colonne', data.columns)
df_filtre = data.loc[id_selected, colonnes_selectionnees] if colonnes_selectionnees else data.loc[id_selected]

if id_selected:
	st.write('ID client sélectionné :', id_selected)
	st.dataframe(df_filtre)

st.subheader("Lancer une simulation de crédit")
go_button = st.button('Go')

if go_button :
	with st.spinner(text='Chargement'):
		shap.initjs()
		url = 'https://kiliandatadev.pythonanywhere.com/prediction'
		
		row_data = data.loc[id_selected].to_dict()
		response = requests.post(url, json=row_data)
		h5 = st.subheader("Resultats")
		st.write(response.text)

		# if response.status_code == 200:
		# 		prediction = response.json()["body"][0]
		# 		st.write('Prediction:', prediction)
		# else:
		# 		st.write('Erreur lors de la requête à l\'API')

		# result = "Crédit autorisé" if prediction == 1 else "Crédit refusé"
		# st.text(result)
		# st.text("Score")

st.subheader("En savoir plus sur la décision du modèle")
shap_button = st.button("Calculer les SHAP values")
if shap_button :
	with st.spinner(text='Chargement'):
		url = 'https://kiliandatadev.pythonanywhere.com/shap-values'
		row_data = data.loc[id_selected].to_dict()
		response = requests.post(url, json=row_data)
		h5 = st.subheader("Resultats")
		st.write(response.text)
		
		# if response.status_code == 200:
		# 		result = response.json()["body"]
		# 		expected_value = result["expected_value"]
		# 		shap_values = result["shap_values"]
		# 		st.write(np.shape(expected_value))
		# 		st.write(np.shape(shap_values))
		# 		st.write(np.shape(shap_values[0][0]))
		# 		st.write(np.shape(data))
				# st.write(shap_values)
				# st.write('data:', data.iloc[0].shape)
		 		# # st.write('shap:', shap_values)
				# df_shap = pd.DataFrame(shap_values[0], columns=data.columns)
				# st.subheader("SHAP values")
				# st.dataframe(df_shap)

				# Ne fonctionne pas
				# st_shap(shap.force_plot(expected_value[0], shap_values[0][0], data.iloc[0,:]))

				# shap.plots.decision(expected_value[0], shap_values[0], data.iloc[1,:], link="logit")
				# shap.force_plot(expected_value[0], shap_values[0][0], data.iloc[1,:], link="logit")
				# shap.force_plot(expected_value[1], df_shap.iloc[1,:], data.iloc[0,:])
				# shap.force_plot(expected_value[0], shap_values[0][0], data.iloc[0,:])

				# Renvoient None
				# shap.multioutput_decision_plot(expected_value, shap_values, row_index=0)
				# shap.summary_plot(np.array(shap_values)[0], data.iloc[0,:])
		# else:
		# 		st.write('Erreur lors de la requête à l\'API')


# Permettre de comparer les informations descriptives relatives à un client à l’ensemble des clients ou à un groupe de clients similaires
# faire une fonction qui trouve les 10 clients similaires (knn) et nous les retourne pour les afficher et pouvoir comparer les variables
st.subheader("Comparer avec des clients similaires")
knn_button = st.button('Rechercher')
if knn_button :
	with st.spinner(text='Chargement'):
		neighbors_df = get_columns_neighbors(id_selected).rename("Clients similaires")
		# mean_df = get_columns_mean().rename("Variables moyennes")
		# st.dataframe(pd.concat([df_filtre[MAIN_COLUMNS], neighbors_df, mean_df], axis=1))

st.write('TODO')
st.write('revoir predict 0/1 et définir un seuil')
# st.write('faire le shap plot') shap value + row
# st.write('faire graph clients similaires') => les trouver par variable cat comme homme femme
# 0 il rembourse 1 il rembourse pas 
