import streamlit as st
import requests

# Fonction pour appeler l'API REST Flask
def predict_tags(sentence):
    url = 'http://127.0.0.1:5000/' 
    response = requests.get(url+'tags/'+sentence)
    if response.status_code == 200:
        return response.json()['tags']
    else:
        return None

# Interface utilisateur Streamlit
st.title('Prédiction de Tags pour une Question')

# Champ de saisie pour le titre de la question
title = st.text_input('Titre de la Question')

# Champ de saisie pour le corps de la question
body = st.text_area('Corps de la Question')

# Bouton pour prédire les tags
if st.button('Prédire les Tags'):
    if title and body:
        sentence = title+' '+body
        tags = predict_tags(sentence)
        if tags:
            st.success(f"Tags prédits: {', '.join(tags)}")
        else:
            st.error("No predicted words")
    else:
        st.warning("Veuillez entrer à la fois le titre et le corps de la question.")

