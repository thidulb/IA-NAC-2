#Thiago Duarte - RM78292

import re
from sklearn.neighbors import KNeighborsClassifier

arquivo = open("iris_dataset.txt", "r") #Abre o arquivo para leitura de dados
conteudoArquivo = arquivo.readlines()
arquivo.close() #Fecha o arqvuio após a leitura dos dados

entrada = []
species = []

regex = r"[,\/]"

for elemento in conteudoArquivo:
    splitElemento = re.split(regex, elemento) #Parametro definido para "quebrar" os itens de uma linha usando regex
    entrada.append([float(splitElemento[0]), float(splitElemento[1]), float(splitElemento[2]), float(splitElemento[3])])
    species.append(str(splitElemento[4]).replace("\n", ""))

knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(entrada, species)

print(knn.predict([[7.0, 3.5, 4.2, 0.5]]))
print(knn.predict_proba([[7.0, 3.5, 4.2, 0.5]]) )

print("\nO Score de medição é de:", knn.score(entrada, species), "(Quanto mais próximo de 1 melhor)")