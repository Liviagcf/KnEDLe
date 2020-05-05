import re
from time import time
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import seaborn as sns
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.gridspec as gridspec
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.metrics import multilabel_confusion_matrix





data = pd.read_csv("train.csv")
df = pd.DataFrame(data)

"""Funcao para pre-processamento dos comentarios:

*   Conversao dos comentarios para letras minusculas;
*   Substituicao de abreviacoes comuns para escrita completa;
*   Remocao de pontuacao.
"""

def cleanText(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text


"""Funcao para plotar Matrix de confusao:"""

def plotConfusionMatrix(y_true, y_pred, classes, title):
    # Calcula acuracia
    acc = accuracy_score(y_true, y_pred)

    plt.figure(figsize=(10,10))

    for i in range(6):
      # Constroi matriz de confusao
      cm = multilabel_confusion_matrix(y_true, y_pred)[i]
      cm = normalize(cm, axis=1, norm='l1')
      cm_df = pd.DataFrame(cm, index = classes, columns = classes)
      plt.subplot(3,3,i+1)
      sns.heatmap(cm_df, annot=True, cmap="YlGnBu")
      plt.title(title[i])
      plt.ylabel('True label')
      plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()



def benchmark():
	pass

def clfTest():
	pass



bin_clf = LinearSVC(loss='l2', penalty='l2',dual=False, tol=1e-3)
clf = OneVsRestClassifier(bin_clf, n_jobs = -1)



df['comment_text'] = df['comment_text'].map(lambda com : cleanText(com))

categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
comment = ['comment_text']

"""Cria o dataframe coom os exemplos rotulados:

*	Seleciona um exemplo para cada rotulo
"""

df_train = pd.DataFrame()

for categorie in categories:
	df_train = df_train.append( df[df[categorie]==1][0:1], ignore_index=True )
	df.drop(index = df[df[categorie]==1][0:1].index, inplace=True)




dataset_serie, _ =  train_test_split(df, random_state=42, test_size=0.0001, shuffle=True)

dataset_serie_comment = dataset_serie.comment_text
dataset_serie_categories = dataset_serie[categories]


"""Extracao dos descritores dos comentarios:"""

vectorizer = TfidfVectorizer(stop_words=stop_words)
comment_vec = vectorizer.fit_transform(dataset_serie_comment)

"""Divisao do dataset entre informacoes de treinamento e teste:"""

tam_data,j = comment_vec.shape
limite_train = round(tam_data*0.66)

data_train = comment_vec[0:limite_train,]
data_test = comment_vec[limite_train:,]
label_train = dataset_serie_categories.iloc[0:limite_train,]
label_test = dataset_serie_categories.iloc[limite_train:,]
