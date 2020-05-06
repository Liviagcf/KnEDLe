import re
from time import time
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
  clf.fit(X_train, y_train)

  confidences = clf.decision_function(X_unlabeled)

  df = pd.DataFrame(clf.predict(X_unlabeled))
  df = df.assign(conf =  confidences.max(1))
  col = categories.copy()
  col.append('conf')
  df.columns = col

  df.sort_values(by=['conf'],ascending = False, inplace = True)
  question_samples = []

  for categorie in categories:
    low_confidence_samples = df[df[categorie]==1].conf.index[0:NUM_QUESTIONS]
    question_samples.extend(low_confidence_samples.tolist())

    df.drop(index = df[df[categorie]==1][0:NUM_QUESTIONS].index, inplace=True)


  return question_samples



def clfTest():
  pred = clf.predict(X_test)

  print("classification report:")
  print(metrics.classification_report(y_test, pred, target_names=categories))

  print("confusion matrix:")
  print(multilabel_confusion_matrix(y_test, pred))


  return metrics.f1_score(y_test, pred, average='micro')


######################################################################################################

categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
comment = ['comment_text']

NUM_QUESTIONS = 2
df = pd.read_csv("train.csv")


bin_clf = LinearSVC(loss='l2', penalty='l2',dual=False, tol=1e-3)
clf = OneVsRestClassifier(bin_clf)#, n_jobs = -1)


"""Aplica a funcao cleanTex em cada comentario:"""
df['comment_text'] = df['comment_text'].map(lambda com : cleanText(com))


"""Divisao do dataset entre informacoes de treinamento e teste:"""

df_test = df.sample(frac = 0.33, random_state = 1)

df_train = df.drop(index = df_test.index)

df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)



"""Cria o dataframe com os exemplos rotulados:

*	Seleciona um exemplo para cada rotulo
"""

df_labeled = pd.DataFrame()

for categorie in categories:
	df_labeled = df_labeled.append( df_train[df_train[categorie]==1][0:1], ignore_index=True )
	df_train.drop(index = df_train[df_train[categorie]==1][0:1].index, inplace=True)

df_unlabeled = df_train


while(True):
  y_train = df_labeled[categories]
  y_test = df_test[categories]

  vectorizer = TfidfVectorizer(encoding= ENCODING, use_idf=True, norm='l2', binary=False, sublinear_tf=True,min_df=0.0001, max_df=1.0, ngram_range=(1, 3), analyzer='word', stop_words=None)

  X_train = vectorizer.fit_transform(df_labeled.comment_text)
  X_test = vectorizer.transform(df_test.comment_text)
  X_unlabeled = vectorizer.transform(df_unlabeled.comment_text)

  df_unified = df_labeled.append(df_unlabeled)
  X_unified  = vectorizer.transform(df_unified.comment_text)


  question_samples = benchmark()
  result_x.append(clf_test())
  result_y.append(df_labeled.toxic.size)




  if df_train.pol.size < 2000:
    insert = {'toxic':[], 'severe_toxic':[], 'obscene':[], 'threat':[], 'insult':[], 'identity_hate':[],'comment_text':[]}
    cont = 0
    for i in question_samples:
      
      try:
        insert["toxic"].insert(cont,df_unlabeled.toxic[i])
        insert["severe_toxic"].insert(cont,df_unlabeled.severe_toxic[i])
        insert["obscene"].insert(cont,df_unlabeled.obscene[i])
        insert["threat"].insert(cont,df_unlabeled.threat[i])
        insert["insult"].insert(cont,df_unlabeled.insult[i])
        insert["identity_hate"].insert(cont,df_unlabeled.identity_hate[i])

        insert["coment_text"].insert(cont,df_unlabeled.text[i])
        cont+=1
        df_unlabeled = df_unlabeled.drop(i)
      except:
        print ("This is an error message!")

    df_insert = pd.DataFrame.from_dict(insert)
    df_train.append(df_insert, ignore_index=True, inplace=True)
    df_unlabeled.reset_index(drop=True, inplace=True)

    #labelNumber = input("Aperte qualquer tecla para proxima iteracao")


  else:
    result_y_active = result_y
    result_x_active = result_x
    plt.plot(result_y_active, result_x_active,label ='Active learning')
    #plt.plot(result_y_spv, result_x_spv,label = 'Convencional')
    plt.axis([0, 2000, 0.3, 0.6])
    plt.legend(loc='lower right', shadow=True, fontsize='x-large')
    plt.grid(True)
    plt.xlabel('Training set size')
    plt.ylabel('f1-score')
    plt.title('Documents set')
    plt.show()

    result = pd.DataFrame(result_y)
    result = result.assign(y=result_x)
    np.savetxt('toxic_results.txt', result, fmt='%f')

    break










'''



# dataset_serie, _ =  train_test_split(df, random_state=42, test_size=0.0001, shuffle=True)

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

'''