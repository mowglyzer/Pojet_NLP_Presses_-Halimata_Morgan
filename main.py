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
print(X_train.shape)
print(y_train.shape)
print(X_test.shape) 
print(y_test.shape)

# Nombre d'articles pour chaque journal  dans l'échantillon cible 
# 0 = Le journal de l'auto, 1 = Le Monde et 2 = Le Télégramme
y_test.value_counts()


#  Important  !    Partie entrainement du modéle à utiliser si on exécute pas le notebook : entrainement sur les données scrapés 

# -------------    A décommenter uniquement si on veut  entrainer le modéle et  créer le script  de classification à partir de ce fichier --------------------


"""

#------   Pipelines pour le preprocessing des données ------------

#Pour le preprocessing , nous utilisons un modèle bag-of-words avec la Pondération selon Tf-Idf 
pipe = make_pipeline(CountVectorizer(), TfidfTransformer())

# ----  Génération de features avec preprocessing-------- 
# à partir du train set(X_train)
pipe.fit(X_train['article'])
X_train_feat = pipe.transform(X_train['article'])
print(X_train_feat.shape)


# Preprocessing de X_test
X_test_feat = pipe.transform(X_test['article'])
print(X_test_feat.shape)


#Sauvegarde du preprecessing pour le traitement des nouveaux articles
with open('mypipe', 'wb') as preprocesseur:
    pickle.dump(pipe, preprocesseur)


# Le modéle choisit pour l'apprentissage  = LogisticRegression
from sklearn.linear_model import LogisticRegression
# Construction du modèle
lr = LogisticRegression()


# --------------------------  Entrainement du modèle  -------------------
lr.fit(X_train_feat, y_train)

print("Score du modéle = ",  lr.score(X_test_feat, y_test) )

# Les prédictions du modéle
y_predict_lr = lr.predict(X_test_feat)

#Test du modèle sur un article
lr.predict(X_test_feat[0])

# Score du modéle pour chaque classe 
precision_score(y_test,lr.predict(X_test_feat), average=None)

# La matrice de confusion des prédictions par LogisticRegression
fig, ax = plt.subplots(figsize=(10, 8))
plot_confusion_matrix(lr, X_test_feat, y_test, cmap=plt.cm.viridis, ax=ax)
plt.show()



#sauvegarde du modèle pour l'utilisation sur des articles inconnus du modèle
with open('logisticReg', 'wb') as model_lr:
    pickle.dump(lr, model_lr)

"""

# ------------------------ Script automatisé ---------------------------------------
def predict(article):
    with open('mypipe', 'rb') as preprocesseur: 
        preprocessing = pickle.load(preprocesseur)
    article_feat = preprocessing.transform([article])
    with open('logisticReg', 'rb') as  model_lr:
        model = pickle.load( model_lr)
    predicted_label = model.predict(article_feat)
    return  predicted_label


# ------     Prédiction sur des articles du test set  d'un article du test set 
#article = data['article'][1]
article = X_test['article'][0]
print(" Source de article[0] = ", predict(article))
# un autre article 
article_1 = X_test['article'][3682]
#article_1
print(" Source de article[3682] = ", predict(article_1))


# Scrappig des articles du jour sur la page internationnal du Monde
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
print("source du nouvel article : ", predict(article_test))


#

#Tous les articles  testés ont été classés correctement !