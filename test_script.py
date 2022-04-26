#Les imports
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import urllib.request as ulib
from urllib.request import Request, urlopen
import bs4
# Import pour le machine learning
import spacy
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import *
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve, GridSearchCV, KFold
import pickle

# Lecture des données scrappés  : Voir le notebook Projet_NLP_presses
data = pd.read_csv("data_Final.csv", index_col=[0])
#print(data)

# # Divise la data en features et target 
#features
X = data[["article"]]
#target 
y =  data['label']
#print(X)

# Séparation des données en test set  et train  set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data[["article"]], data['label'], test_size=0.25, random_state=42)

#shape des split
print(X_train.shape)
print(y_train.shape)
print(X_test.shape) 
print(y_test.shape)

# Nombre d'articles pour chaque journal  dans l'échantillon cible 
# 0 = Le journal de l'auto, 1 = Le Monde et 2 = Le Télégramme
print(y_test.value_counts())


# Le script  pour la prédiction des sources des articles . testé ds le notebook  


def predict(article):
    with open('mypipe', 'rb') as preprocesseur: 
        preprocess = pickle.load(preprocesseur)
    article_feat = preprocess.transform([article])
    with open('logisticReg', 'rb') as  model_lr:
        model = pickle.load( model_lr)
    predicted_label = model.predict(article_feat)
    return  predicted_label


# Prediction de la source d'un article test
article_1 = X_test['article'][3733] # article du journal de l'auto
print("source de article_1 : ", predict(article_1))



# Scrappig des articles du jour(28/03/2022) sur la page internationnal du Monde pour tester la classification de nouvel article
links=[]
url= 'https://www.lemonde.fr/international/' 
req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
html = urlopen(req).read()
page = bs4.BeautifulSoup(html, 'html.parser')

for i in  page.findAll('div', {'class' : 'thread'}):
    links.append(i.find('a').get("href"))
        
list_test=[]
for link in links:
    req = Request(link, headers={'User-Agent': 'Mozilla/5.0'})
    html = urlopen(req).read()
    page = bs4.BeautifulSoup(html, 'html.parser')

    title = page.find('h1').getText()
    article=""
    for t in  page.findAll("p", {'class': "article__paragraph"}):
        article += t.text.replace('\xa0','') 
        article = article.replace('\n', '')
        article = article.replace('\t', '')
    length=len(article.split())
    if length < 20:
        print(title, length)
    elif length >=20:
        list_test.append((article))

# Prédiction de la source de l'article
article_test =  list_test[0]
print("source du nouvel article : ", predict(article_test)) # article bien classé 