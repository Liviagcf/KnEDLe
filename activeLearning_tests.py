import re
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt


def cleanText(text):
    '''Normalização do texto retirando acentuação, caracteres especiais,
       espaços adicionais e caracteres não textuais'''

    text = text.lower()
    text = re.sub(r"ú", "u", text)
    text = re.sub(r"á", "a", text)
    text = re.sub(r"é", "e", text)
    text = re.sub(r"í", "i", text)
    text = re.sub(r"ó", "o", text)
    text = re.sub(r"u", "u", text)
    text = re.sub(r"â", "a", text)
    text = re.sub(r"ê", "e", text)
    text = re.sub(r"ô", "o", text)
    text = re.sub(r"à", "a", text)
    text = re.sub(r"ã", "a", text)
    text = re.sub(r"õ", "o", text)
    text = re.sub(r"ç", "c", text)
    text = re.sub(r"\\W", " ", text)
    text = re.sub(r"\\s+", " ", text)
    text = text.strip(' ')
    return text


clf = MultinomialNB()
clfSVM = LinearSVC(loss='squared_hinge', penalty='l2', dual=False, tol=1e-3)

categories = [
    'secretaria de estado de seguranca publica',
    'secretaria de estado de cultura',
    'secretaria de estado de fazenda planejamento orcamento e gestao',
    'casa civil',
    'secretaria de estado de obras e infraestrutura',
    'secretaria de estado de educacao',
    'defensoria publica do distrito federal',
    'secretaria de estado de saude',
    'tribunal de contas do distrito federal',
    'secretaria de estado de desenvolvimento urbano e habitacao',
    'poder legislativo',
    'secretaria de estado de justica e cidadania',
    'secretaria de estado de transporte e mobilidade',
    'controladoria geral do distrito federal',
    'poder executivo',
    'secretaria de estado de agricultura abastecimento e desenvolvimento rural',
    'secretaria de estado de economia desenvolvimento inovacao ciencia e tecnologia',
    'secretaria de estado de desenvolvimento economico',
    'secretaria de estado do meio ambiente']

PATH_TRAIN = "dodftrain.csv"
ENCODING = 'utf-8'

df = pd.read_csv(PATH_TRAIN,encoding = ENCODING,header = 0)
df['label'] = df['label'].map(lambda com: cleanText(com))
df['text'] = df['text'].map(lambda com: cleanText(com))

"""Divisao do dataset entre informacoes de treinamento e teste:"""

# df_test = df.sample(frac = 0.33, random_state = 1)
df_test = df.sample(n=50, random_state = 1)

df_train = df.drop(index = df_test.index)

df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)


y_train = df_train.label
y_test = df_test.label

vectorizer = TfidfVectorizer(encoding= ENCODING, use_idf=True, norm='l2', binary=False, sublinear_tf=True,min_df=0.0001, max_df=1.0, ngram_range=(1, 3), analyzer='word', stop_words=None)

X_train = vectorizer.fit_transform(df_train.text)
X_test = vectorizer.transform(df_test.text)

clf.fit(X_train, y_train)
clfSVM.fit(X_train, y_train)

# confidences = clf.predict_log_proba(X_test)
# print('Confidences: ', confidences[0])
# print('Categories: ', len(categories))
print('Acuracy NB: ', clf.score(X_test, y_test))
print('Acuracy SVM: ', clfSVM.score(X_test, y_test))
