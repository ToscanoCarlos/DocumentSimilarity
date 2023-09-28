#Extract from the news corpus the news section and generate a new corpus with this content
#-Tokenize the extracted content and create a tokenized corpus
#-Lemmatize the corpus
#-Remove stop words that are articles,prepositions,conjunctions and pronouns

import re
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import math 


def cosine(x, y):
	val = sum(x[index] * y[index] for index in range(len(x)))
	sr_x = math.sqrt(sum(x_val**2 for x_val in x))
	sr_y = math.sqrt(sum(y_val**2 for y_val in y))
	res = val/(sr_x*sr_y)
	return (res)


def lematizarPrueba(prueba):
        
    # Cargar el modelo de spaCy en español
    nlp = spacy.load("es_core_news_sm")

    # String que quieres procesar
    #save the news sections in a file
    #with open('corpus_noticias_secciones.txt', 'w', encoding='utf-8') as archivo:
     #   prueba = archivo.read()

    # TOKENIZAR
    doc = nlp(prueba)

    # Obtener tokens del documento
    tokens = [token.text for token in doc]

    # LEMATIZAR
    lemmatized_tokens = []

    # Convertir los tokens en un texto
    prueba_text = " ".join(tokens)

    # Procesar el texto con SpaCy para lematizar
    doc = nlp(prueba_text)

    # Obtener lemas de los tokens
    lemmas = [token.lemma_ for token in doc]

    # Agregar los lemas a 'lemmatized_tokens'
    lemmatized_tokens.extend(lemmas)
    stopwords  = [
        "el", "la", "los", "las", "un", "una", "unos", "unas","la","lo","a","al","del"  # Artículos
        "a", "ante", "bajo", "con", "contra", "de", "desde", "en", "entre", "hacia", "para", "por", "según", "sin", "sobre", "tras",  # Preposiciones
        "y", "o", "pero", "ni", "que", "si", "como", "porque","u",  # Conjunciones
        "yo", "tú", "él", "ella", "nosotros", "vosotros", "ellos", "ellas", "mí", "ti", "sí", "nos", "vos", "se", "me", "te", "le", "nos", "os", "les", "se",  # Pronombres personales
    ]
    # ELIMINAR STOPWORDS
    filtered_tokens = [token for token in lemmatized_tokens if token.lower() not in stopwords]

    # Imprimir la lista de tokens lematizados sin stopwords
    #print(filtered_tokens)
    texto = " ".join(filtered_tokens)
    print(texto)
    return texto

#leer el archivo de noticias y guardar en un arreglo
noticias = []
#leer archivo y guardar en arreglo donde cada salto de linea es un elemento
#El bueno es corppus_normalizado.txt, corpus_final.txt para probar
with open('corpus_final.txt', 'r', encoding='utf-8') as archivo:
    lineas = archivo.readlines()
    for linea in lineas:
        noticias.append(linea.strip('\n'))

#vectorizar el arreglo de noticias
vectorizador_binario = CountVectorizer(binary=True)
xBinario = vectorizador_binario.fit_transform(noticias)

vectorizador_frecuencia = CountVectorizer(token_pattern= r'(?u)\w\w+|\w\w+\n|\.')
xFrecuencia = vectorizador_frecuencia.fit_transform(noticias)

vectorizador_tfidf = TfidfVectorizer(token_pattern= r'(?u)\w\w+|\w\w+\n|\.')
xtfidf = vectorizador_tfidf.fit_transform(noticias)

#leer el archivo de prueba y guardar en un arreglo
with open('./pruebas noticia/pruebajeje.txt', 'r', encoding='utf-8') as archivo:
    pruebatxt = archivo.read()

pruebaNoticia = [lematizarPrueba(pruebatxt)]

yBinario = vectorizador_binario.transform(pruebaNoticia)

yFrecuencia = vectorizador_frecuencia.transform(pruebaNoticia)

ytfidf = vectorizador_tfidf.transform(pruebaNoticia)

xArreglo = xFrecuencia.toarray()
yArreglo = yFrecuencia.toarray()

#recorrer el arreglo de noticias y comparar con la noticia de prueba
#guardar los resultados en un arreglo
similitud = []
for i in range(len(xArreglo)):
    similitud.append(cosine(xArreglo[i],yArreglo[0]))

#mostrar el top 10 de noticias mas similares
print("Top 10 noticias mas similares")
for i in range(10):
    index = similitud.index(max(similitud))
    print(i+1, "Noticia: ", index, "Similitud: ", similitud[index])
    similitud[index] = 0
















#similitud = cosine(xArreglo[0],yArreglo[0])#,yArreglo[0]
#print(similitud)


# cosine(xBinario.toarray(), yBinario.toarray())
# cosine(xFrecuencia[0], yFrecuencia.toarray())
# cosine(xtfidf[0], ytfidf.toarray())

# print (vectorizador_binario.get_feature_names_out())

# similarities = cosine_similarity(xFrecuencia, yFrecuencia)
# most_similar_indices = similarities.argsort(axis=1)[:, ::-1]

#similarities = cosine_similarity(xtfidf, ytfidf)
#most_similar_indices = similarities.argsort(axis=1)[:, ::-1]

# Mostrar el "top n" de resultados, por ejemplo, los 5 vectores más similares
#top_n = 10
#for i in range(top_n):
#    index = most_similar_indices[0][i]  # Obtener el índice del i-ésimo vector más similar
#    similarity_score = similarities[0][index]  # Obtener la puntuación de similitud coseno correspondiente
#    print(f"Documento más similar {i + 1}: Índice {index}, Similitud Coseno: {similarity_score}")
# La variable 'similarity' ahora contiene la similitud coseno entre todos los pares de vectores en xFrecuencia y yFrecuencia.
# Si quieres obtener la similitud entre el primer vector de xFrecuencia y el primer vector de yFrecuencia, puedes acceder a ella de la siguiente manera:
