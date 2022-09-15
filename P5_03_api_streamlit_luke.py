# -*- coding: utf-8 -*-
"""
Created on Mon May 16 08:41:16 2022

@author: luked
"""
#%%
# =============================================================================
# Variables initiales à modifier
# =============================================================================
# Booléen d'activation (ou non) de l'installation des modules/librairies nécessaires au bon fonctionnement de ce Notebook (cf partie 1.1) du Notebook).
installation = False

### Tout ce qui concerne les entrées de ce notebook
abs_path_cache_input_donnees = "../Cache_fichiers/Analyse/V04test_"#"C:/Users/luked/Documents/Formation_Ingenieur_ML/Projets/P5/Cache_fichiers/Analyse/V04test_"
abs_path_cache_input_modeles = "../Cache_fichiers/TestModeles/V04test_"##"C:/Users/luked/Documents/Formation_Ingenieur_ML/Projets/P5/Cache_fichiers/TestModeles/V04test_"
abs_path_cache_input_self = "../Cache_fichiers/CodeFinalAPI/V04test_"##"C:/Users/luked/Documents/Formation_Ingenieur_ML/Projets/P5/Cache_fichiers/CodeFinalAPI/V04test_"
### Tout ce qui concerne les figures
# Sauvegarde des figures et chemin de données vers répertoire de stockage
sauvegarde_figure, abs_path_fig = (
    True,
    "../Figures/CodeFinalAPI/V04test_",#"C:/Users/luked/Documents/Formation_Ingenieur_ML/Projets/P5/Figures/CodeFinalAPI/V04test_",
)

### Tout ce qui concerne le cache des fichiers (pour ne pas avoir à refaire tous les calculs systématiquement)
# Sauvegarde des figures et chemin de données vers répertoire de stockage
sauvegarde_fichiers, abs_path_cache_output = (
    True,
    "../Cache_fichiers/CodeFinalAPI/V04test_",#"C:/Users/luked/Documents/Formation_Ingenieur_ML/Projets/P5/Cache_fichiers/CodeFinalAPI/V04test_",
)

### Autres
# Booléens d'activation du test de bon chargement du modèles "optimal" choisi
tester_modele = False

#%%
# =============================================================================
# Installations
# =============================================================================
# Installation conditionnée au booléen précédent
#if installation:
    #!pip install tensorflow
    #!pip install tensorflow_hub
    #!pip install tensorflow_text
    #!pip install streamlit
    
#%%
# =============================================================================
# Importations
# =============================================================================
### Quelques classiques
import numpy as np
import pandas as pd
#import tqdm
### Traitement de texte
import re
#import gensim
### Scikit-learn
#from sklearn.model_selection import train_test_split#, StratifiedKFold,cross_validate, 
### Tensorflow
#import tensorflow as tf
#from tensorflow.keras import backend as K
#from tensorflow.keras.preprocessing.text import Tokenizer
#from tensorflow.keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras import Input, Model
#from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D
import tensorflow_hub as hub
import tensorflow_text
#import transformers
# picle
import pickle
import streamlit as st

#%%
# =============================================================================
# constantes et Fonctions personnelles
# =============================================================================
# Random State fixé une bonne fois pour toute
rgn = 420
tags_with_W = pd.read_csv(abs_path_cache_input_donnees + 'tags_with_W.csv', index_col=0)
#embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def tokens_to_string(list_of_tokens):
    string = list_of_tokens[0]
    for n in range(1,len(list_of_tokens)):
        string += ' '+list_of_tokens[n]
    return string

def pickle_load(path_and_title):
    file_pickel_rb = open(file=path_and_title,mode='rb')
    objet = pickle.load(file_pickel_rb)
    file_pickel_rb.close()
    del file_pickel_rb
    return objet
   
#%%
# =============================================================================
# 
# =============================================================================
st.title('OpenClassroom - Machine learning engineer cursus - Categorize question automatically (n°5 project)\n by Luke Duthoit')

### On redéfinit certains paramètres nécessaires à l'extraction de features par le modèle USE
# embede contient le chargement d'un modèle prédéfini
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
# chargement du modèle "optimal" basé sur des features produites par USE
model = pickle_load(abs_path_cache_input_modeles+'moc_logreg_use')

### Fonction de production de features
def feature_USE_fct(sentences, b_size) :
    batch_size = b_size
    for step in range(len(sentences)//batch_size) :
        idx = step*batch_size
        feat = embed(sentences[idx:idx+batch_size])
        if step ==0 :
            features = feat
        else :
            features = np.concatenate((features,feat))
    ### Lignes supplémenatires, par Luke
    ### Si le nombre de documents n'est pas divisible par la taille du batch, on reproduit la boucle précédente sur les documents restants
    if (len(sentences)//batch_size)*batch_size != len(sentences) :
        feat = embed(sentences[idx+batch_size:])
        features = np.concatenate((features,feat))
    ### Fin des lignes supplémenatires    
    return features

### Fonction de transformation d'une chaîne de caractère
def transform_text(input_text='bla bla bla'):
    output_text = input_text.lower()
    output_text = re.split(string=output_text, pattern=' ')
    output_text = [[w] if w in tags_with_W.values else re.findall(string=w, pattern='[a-z]+') for w in output_text]
    output_text = np.concatenate(output_text)
    output_text = tokens_to_string(output_text)#.apply(lambda x : tokens_to_string(x)).values
    return(output_text)

def predire_tags_uniques(model, features):
    tab_pred_all_doc = []
    predictions = model.predict(features)
    for i, pred in enumerate(predictions) :
        uniques, index = np.unique(pred, return_index=True)
        if len(pred) == len(uniques) :
            tab_pred_all_doc.append(pred) 
        else :
            classes, probas = model.classes_, model.predict_proba(features)
            nouv_pred = ['', '', '', '', '']
            for n in index :
                if nouv_pred[n] == '':
                    nouv_pred[n] = uniques[index==n][0]
            for n in range(5):
                if nouv_pred[n] == '' :
                    possibilites = classes[n][np.argsort(probas[n][i])[-1:-6:-1]]
                    j, tag = 0, possibilites[0]
                    while tag in nouv_pred :#uniques :
                        j +=1
                        tag = possibilites[j]
                    nouv_pred[n]= tag
            tab_pred_all_doc.append(nouv_pred)
            del nouv_pred, classes, probas, n, possibilites, j, tag
        del uniques, index
    del i, pred
    return tab_pred_all_doc

### Fonction de prédiction
def my_prediction(input_text='Bla bla 1.bla ?'):
    output_text = transform_text(input_text)
    # reformatage des données d'entrée
    sentences = np.asarray([output_text]).reshape(-1)
    # passage des chaînes de caractères aux features
    features = feature_USE_fct(sentences=sentences, b_size=min(100,max(1,len(sentences)//10)))
    # prediction
    pred = predire_tags_uniques(model, features)[0]#model.predict(features)[0]
    return pred

question_0 = 'How can i serialize a javascript api with streamlit if streamlit is in python-3.x ?'

question_input = st.text_input('Write a question in the stackoverflow`s style :', question_0)
st.write('Our model suggests the following tags', my_prediction(question_input))