#Thiago Duarte - RM78292

import re
from sklearn import preprocessing
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
import numpy as np

arquivo = open("dados.txt", "r") #Abre o arquivo para leitura de dados
conteudoArquivo = arquivo.readlines()
arquivo.close() #Fecha o arqvuio após a leitura dos dados

np.set_printoptions(threshold=np.inf)

Casa = []
Civil = []
Rendimento = []
Resultado = []
regex = r"\s"

for elemento in conteudoArquivo:
    splitElemento = re.split(regex, elemento) #Parametro definido para "quebrar" os itens de uma linha usando regex
    Casa.append(splitElemento[0])
    Civil.append(splitElemento[1])
    Rendimento.append(splitElemento[2])
    Resultado.append(splitElemento[3])

le = preprocessing.LabelEncoder()

casa_numero = le.fit_transform(Casa)
# print(casa_numero)
civil_numero = le.fit_transform(Civil)
# print(civil_numero)
rendimento_numero = le.fit_transform(Rendimento)
# print(rendimento_numero)
resultado_numero = le.fit_transform(Resultado)
# print(resultado_numero)

entrada_dados=zip(casa_numero, civil_numero, rendimento_numero)
entrada_dados = tuple(entrada_dados) #Usar a funçao tuple() para deixar o resultado "legível"
# print(entrada_dados)
clf = GaussianNB()
clf.fit(entrada_dados, resultado_numero)
print(clf.predict(entrada_dados))

print(clf.score(entrada_dados, resultado_numero))
print(clf.predict_proba(entrada_dados))