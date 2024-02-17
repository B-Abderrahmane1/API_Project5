from flask import Flask
from flask_restful import Api, Resource
import pandas as pd
from nltk.tokenize import word_tokenize
import pickle as pc
import tensorflow_hub as hub
import numpy

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
batch_size = 8
labels = pd.read_csv('labels.csv')

###################################################################################
##Functions
###################################################################################
def tokenizer_fct(sentence) :
    sentence_clean = sentence.replace('-', ' ').replace('/', ' ')
    word_tokens = word_tokenize(sentence_clean)
    return word_tokens
    
def lower_start_fct(list_words) :
    lw = [w.lower() for w in list_words if (not w.startswith("@"))                                  
                                       and (not w.startswith("http"))]
    return lw
    
def transform_dl_fct(desc_text) :
    word_tokens = tokenizer_fct(desc_text)
    lw = lower_start_fct(word_tokens)    
    transf_desc_text = ' '.join(lw)
    return transf_desc_text

def feature_USE_fct(sentences, b_size) :
    batch_size = b_size
    for step in range(len(sentences)//batch_size) :
        idx = step*batch_size
        feat = embed(sentences[idx:idx+batch_size])
        if step ==0 :
            features = feat
        else :
            features = np.concatenate((features,feat))
    return features
###################################################################################

app = Flask(__name__)
api = Api(app)

filename = 'finalized_model.sav'
loaded_model = pc.load(open(filename, 'rb'))

class Tags(Resource):
    def get(self, sentence):
        doc = transform_dl_fct(sentence)
        pre_features= pd.Series([doc]*batch_size)
        features = feature_USE_fct(pre_features, batch_size)
        y_pred = loaded_model.predict(features)[0]
        arr = numpy.nonzero(y_pred)
        y_label = labels['0'][arr[0]]

        return {"tags": y_label.tolist()}

api.add_resource(Tags, "/tags/<string:sentence>")

if __name__ == "__main__":
    app.run(debug=True)



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

