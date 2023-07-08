import streamlit as st
import pandas as pd
import numpy as np
import requests

st.title('Prêt à dépenser : Scoring Crédit')
st.image('./assets/images/logo-app.png', width=250)

h1 = st.subheader('Tableau de données de nos clients')
data = pd.read_csv('./data/X_train_sample.csv', index_col=[0])
st.dataframe(data)

h2 = st.subheader("Sélectionner un client")
ids_index = data.index

id_selected = st.selectbox('Sélectionner un ID', options=ids_index, index=0)
colonnes_selectionnees = st.multiselect('Filtrer par colonne', data.columns)
# Permettre de visualiser des informations descriptives relatives à un client (via un système de filtre)
df_filtre = data.loc[id_selected, colonnes_selectionnees] if colonnes_selectionnees else data.loc[id_selected]

if id_selected:
	st.write('ID client sélectionné :', id_selected)
	st.dataframe(df_filtre)

h4 = st.subheader("Lancer une simulation de crédit")
go_button = st.button('Go')

# Permettre de visualiser le score et l’interprétation de ce score pour chaque client de façon intelligible pour une personne non experte en data science
if go_button :
	with st.spinner(text='Chargement'):
		
		# row_df = row_df = pd.DataFrame([row_data], index=[id_selected])
		# st.write(row_df)	
		# st.dataframe(row_df)

		url = 'http://pitiprince.pythonanywhere.com/data'
		
		row_data = data.loc[id_selected].to_dict()
		response = requests.post(url, json=row_data)
		h5 = st.subheader("Resultats")
		st.write(response)
		st.write(response.text)

		if response.status_code == 200:
				prediction = response.json()['body']
				st.write('prediction:', prediction)
		else:
				st.write('Erreur lors de la requête à l\'API')
				# st.write(response.json()['error'])
		# if response.status_code == 200 and 'body' in response.json():
		# prediction = response.text
		# st.write('prediction:', prediction)
		# 	# Faites quelque chose avec la prédiction
		# # else:
		# 	# st.write(response.json().get('error'))

		# st.text("Granted or Refused")
		# st.text("Score")

st.subheader("Display SHAP values")
# st.line_chart(data.iloc[0,:])

# Permettre de comparer les informations descriptives relatives à un client à l’ensemble des clients ou à un groupe de clients similaires
# faire une fonction qui trouve les 10 clients similaires (knn) et nous les retourne pour les afficher et pouvoir comparer les variables
h7 = st.subheader("Display similar users")
data = np.random.randint(1, 101, size=(10, 20))
df2 = pd.DataFrame(data, columns=[f"col {i}" for i in range(20)])