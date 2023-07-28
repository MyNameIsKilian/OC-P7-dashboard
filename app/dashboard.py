import streamlit as st
import pandas as pd
import numpy as np
import requests
import shap
import json
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import os

# Exécuter le script setup.sh pour installer les dépendances
os.system("../setup.sh")

np.__version__ = "1.23"

TIMEOUT = (5, 30)
# MAIN_COLUMNS = ['CODE_GENDER_M', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
#                 'NAME_FAMILY_STATUS_Married', 'NAME_INCOME_TYPE_Working',
#                 'AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED']

MAIN_COLUMNS = ['CNT_CHILDREN','APPS_EXT_SOURCE_MEAN', 'APPS_GOODS_CREDIT_RATIO',
                'AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED']

API_URL = "https://kiliandatadev.pythonanywhere.com/"

# class ShapObject:
#     def __init__(self, base_values, data, values, feature_names, display_data):
#         self.base_values = base_values # Single value
#         self.data = data # Raw feature values for 1 row of data
#         self.values = values # SHAP values for the same row of data
#         self.feature_names = feature_names # Column names
#         self.display_data = display_data

def st_shap(plot, height=None):
    """ Create a shap html component """
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def get_columns_mean():
    """ Get customers main columns mean values """
    response = requests.get(API_URL + "columns/mean", timeout=TIMEOUT)
    content = response.json()
    return pd.Series(content)
 
def get_columns_neighbors(cust_id):
    """ Get customers neighbors main columns mean values """
    response = requests.get("https://kiliandatadev.pythonanywhere.com/columns/neighbors/id/" + str(cust_id))
    content = response.json()
    return pd.Series(content)

st.title('Prêt à dépenser : Scoring Crédit')
st.image('./assets/images/logo-app.png', width=250)

st.subheader('Tableau de données de nos clients')
data = pd.read_csv('./data/X_train_sample.csv', index_col=[0])
st.dataframe(data)

st.subheader("Sélectionner un client")
ids_index = data.index

id_selected = st.selectbox('Sélectionner un ID', options=ids_index, index=0)
iloc_id_selected = ids_index.get_loc(id_selected)
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
		# 0 il rembourse 1 il rembourse pas 

		if response.status_code == 200:
				prediction = response.json()["body"][0]
				proba_authorized = prediction[0]
				result = "Crédit autorisé" if proba_authorized > 0.5 else "Crédit refusé"
				st.text("Les crédits sont autorisés si votre score client est supérieur à 50 points")
				st.text("Score : " + str(round(proba_authorized * 100,1)) + "/100 points")
				st.text(result)
		else:
				st.write('Erreur lors de la requête à l\'API')

st.subheader("En savoir plus sur ce résulat")
shap_button = st.button("Calculer les SHAP values")
if shap_button :
	with st.spinner(text='Chargement'):
		url = 'https://kiliandatadev.pythonanywhere.com/shap-values'
		row_data = data.loc[id_selected].to_dict()
		response = requests.post(url, json=row_data)
		h5 = st.subheader("Resultats")
		
		if response.status_code == 200:
			result = response.json()["body"]
			expected_value = result["expected_value"]
			shap_values_cleaned = np.array(result["shap_values"])
			using_shap_plots = False
			if using_shap_plots:
				shap_values_full = json.loads(result["shap_values_full"])
				shap_values = shap_values_full["values"]
				st.write(np.shape(expected_value))
				st.write(np.shape(shap_values_cleaned))
				st.write('shape:', np.array(shap_values_cleaned).shape)
				st.write(np.shape(shap_values[0][0]))
				st.write(np.shape(data))
				st.write(shap_values)
				st.write(shap_values_cleaned)
				st.write('data:', data.iloc[0].shape)
				st.write('shap:', shap_values_cleaned)
				st.write('expec:', expected_value)

				# shap_object = ShapObject(base_values = expected_value[1],
				#          values = shap_values_cleaned[1][0,:],
				#          feature_names = data.columns,
				#          data = data.iloc[0,:],
				# 		 display_data = data.iloc[0,:])

				# shap_input = ShapObject(expected_value[0], shap_values_cleaned[0], 
				#        data.iloc[0,:], feature_names=data.columns, display_data=data.iloc[0,:])

				# shap.waterfall_plot(shap_input)

				# shap.plots.waterfall(shap_object)
				
				# shap.waterfall_plot(shap_object)

				# st_shap(shap.plots.decision(expected_value[1], shap_values_cleaned[1], row_scaled, link="logit"))

				# shap_explanation = shap.Explanation(shap_values, base_values=expected_value, data=shap_values_full["data"])
				# _____________________________________________________________________________________________
				# Waterfall plots
				# st_shap(shap.plots._waterfall.waterfall_legacy(expected_value[0], np.array(shap_values_cleaned)))
				# IndexError: index 212 is out of bounds for axis 0 with size 2

				# st_shap(shap.plots.waterfall(expected_value, shap_values))
				# base_values = shap_values.base_values
				# AttributeError: 'list' object has no attribute 'base_values'

				# st_shap(shap.waterfall_plot(expected_value, shap_values))
				# AttributeError: 'list' object has no attribute 'base_values'

				# st_shap(shap.waterfall(shap_values))
				# AttributeError: module 'shap' has no attribute 'waterfall'
				# _____________________________________________________________________________________________
				# Other plots
				# st_shap(shap.force_plot(expected_value[0], shap_values_cleaned[0][0], data.iloc[0,:]))
				# AssertionError: The shap_values arg looks multi output, try shap_values[i] avec shap_values_cleaned[0][0]
				# AssertionError: visualize() can only display Explanation objects (or arrays of them) avec shap_values_cleaned[0][0][0]

				# st_shap(shap.force_plot(shap_explanation))
				# Exception: In v0.20 force_plot now requires the base value as the first parameter! Try shap.force_plot(explainer.expected_value, shap_values) 
				# or for multi-output models try shap.force_plot(explainer.expected_value[0], shap_values[0]).

				# shap.plots.decision(expected_value[0], shap_values_cleaned[0], data.iloc[1,:], link="logit")
				# TypeError: Looks like multi output. Try base_value[i] and shap_values[i], or use shap.multioutput_decision_plot().

				# Renvoient None
				# st_shap(shap.multioutput_decision_plot(expected_value, shap_values_cleaned, row_index=0))

				# shap.summary_plot(np.array(shap_values_cleaned), data.iloc[0,:])
				# IndexError: index 212 is out of bounds for axis 1 with size 1

				# shap.plots.bar(shap_values)

			df_feature_importance_class_0 = pd.DataFrame(data=shap_values_cleaned[0], index=[id_selected], columns=data.columns)
			df_user_feat_imp = df_feature_importance_class_0.iloc[0, :]
			
			top3_class_1 = df_user_feat_imp.sort_values(ascending=False).head(3)
			top3_class_0 = df_user_feat_imp.sort_values(ascending=True).head(3)
			# st.write(top3_class_0)
			# st.write(top3_class_1)
			
			col1, col2 = st.columns(2)
			with col1:
				fig = plt.Figure(figsize=(4, 4))
				ax = fig.subplots()
				ax.pie(top3_class_0, labels = top3_class_0.index, autopct='%.0f%%')
				st.pyplot(fig)
			# with col2:
			# 	fig = plt.Figure(figsize=(4, 4))
			# 	ax = fig.subplots()
			# 	ax.pie(top3_class_1, labels = top3_class_1.index, autopct='%.0f%%')
			# 	st.pyplot(fig)
		else:
				st.write('Erreur lors de la requête à l\'API')

st.subheader("Comparer avec des clients similaires")
knn_button = st.button('Rechercher')
if knn_button :
	with st.spinner(text='Chargement'):
		customer_df = data[MAIN_COLUMNS].loc[id_selected]
		neighbors_df = get_columns_neighbors(iloc_id_selected).rename("Moyennes des clients similaires")
		mean_df = get_columns_mean().rename("Moyennes de tous les clients")
		concat_df = pd.concat([customer_df, neighbors_df, mean_df], axis=1)
		st.dataframe(concat_df)
		# selected_column = st.selectbox('Choisir une colonne', MAIN_COLUMNS)
		for col in MAIN_COLUMNS:
			df_for_chart = concat_df.loc[[col]] 
			df_for_chart = df_for_chart.transpose()
			df_for_chart.columns = [col]
			st.bar_chart(df_for_chart)

		# Point à revoir pendant la session:
		# - Les shap values (valeurs négatives + compréhension sort values)
		# - Tips pour que tous les éléments du dashboard restent en place

		# faire le lien github pythonanywhere  => webhook ?
		# faire de la documentation
		# commencer support de présentation


