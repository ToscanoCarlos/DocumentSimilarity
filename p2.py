#Extract from the news corpus the news section and generate a new corpus with this content
#-Tokenize the extracted content and create a tokenized corpus
#-Lemmatize the corpus
#-Remove stop words that are articles,prepositions,conjunctions and pronouns

import re
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import math 


def cosine(x, y):
	val = sum(x[index] * y[index] for index in range(len(x)))
	sr_x = math.sqrt(sum(x_val**2 for x_val in x))
	sr_y = math.sqrt(sum(y_val**2 for y_val in y))
	res = val/(sr_x*sr_y)
	return (res)

#Funcion para tokenizar lematizar y todoo eso
def lematizarCorpusOriginal():
        
    # Use regex to extract the news section between "&&&&&" delimiters
    expresion  =  r'([^&]+)&&&&&&&&'
    news_sections = re.findall(expresion, corpus_original)

    news_sections = news_sections[2::3]

    """#save the news sections in a file
    with open('corpus_noticias_secciones.txt', 'w', encoding='utf-8') as archivo:
        for news in news_sections:
            archivo.write(news + '\n')"""


    # Cargar el modelo de spaCy en español
    nlp = spacy.load("es_core_news_sm")

    # TOKENIZAR
    tokenized_sections = []

    for section in news_sections:
        # Procesar el texto con SpaCy
        doc = nlp(section)
        
        # Obtener tokens del documento
        tokens = [token.text for token in doc]
        
        # Agregar la lista de tokens a 'tokenized_sections'
        tokenized_sections.append(tokens)

    # Lemmatizar


    lemmatized_sections = []

    for section_tokens in tokenized_sections:
        section_text = " ".join(section_tokens)
        doc = nlp(section_text)
        lemmatized_tokens = [token.lemma_ for token in doc]
        lemmatized_sections.append(lemmatized_tokens)

    # Remove stop words that are articles,prepositions,conjunctions and pronouns en español
    # Agregar signos de puntuación a la lista de stopwords
    #punctuation = ['.', ',', ';', '!', '?', ':', '-', '"', '(', ')', '[', ']', '{', '}', '...', '¡', '¿', '»', '«', '...', '``', "''", '/', '|', '“', '”','*','él']

    stopwords  = [
        "el", "la", "los", "las", "un", "una", "unos", "unas","la","lo","a","al","del"  # Artículos
        "a", "ante", "bajo", "con", "contra", "de", "desde", "en", "entre", "hacia", "para", "por", "según", "sin", "sobre", "tras",  # Preposiciones
        "y", "o", "pero", "ni", "que", "si", "como", "porque","u",  # Conjunciones
        "yo", "tú", "él", "ella", "nosotros", "vosotros", "ellos", "ellas", "mí", "ti", "sí", "nos", "vos", "se", "me", "te", "le", "nos", "os", "les", "se",  # Pronombres personales
    ]

    #stopwords = stopwords.union(punctuation)#spacy.lang.es.stop_words.STOP_WORDS.union(punctuation)

    # Eliminar stopwords de cada lista de tokens lematizados
    filtered_sections = []

    for lemmatized_tokens in lemmatized_sections:
        filtered_tokens = [token for token in lemmatized_tokens if token.lower() not in stopwords]
        filtered_sections.append(filtered_tokens)

    # Save the filtered corpus in a file
    with open('corpus_final.txt', 'w', encoding='utf-8') as archivo:
        for section in filtered_sections:
            section_text = " ".join(section)
            archivo.write(section_text + '\n')
    







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
    print(filtered_tokens)

    




# read corpus from file
# Abre el archivo y lee su contenido
with open('corpus_noticias.txt', 'r', encoding='utf-8') as archivo:
    corpus_original = archivo.read()

#lematizarCorpusOriginal(corpus_original)

noticias = []
#leer archivo y guardar en arreglo donde cada salto de linea es un elemento
with open('corpus_final.txt', 'r', encoding='utf-8') as archivo:
    lineas = archivo.readlines()
    for linea in lineas:
        noticias.append(linea.strip('\n'))

#print (noticias)
#Para el de prueba solo vectorizador_binario.transform
vectorizador_binario = CountVectorizer(binary=True)
xBinario = vectorizador_binario.fit_transform(noticias)
print ('Representación vectorial binarizada')
print (xBinario.toarray())

vectorizador_frecuencia = CountVectorizer(token_pattern= r'(?u)\w\w+|\w\w+\n|\.')
xFrecuencia = vectorizador_frecuencia.fit_transform(noticias)
print('Representación vectorial por frecuencia')
print (xFrecuencia.toarray())

vectorizador_tfidf = TfidfVectorizer(token_pattern= r'(?u)\w\w+|\w\w+\n|\.')
xtfidf = vectorizador_tfidf.fit_transform(noticias)
print ('Representación vectorial tf-idf')
print (xtfidf.toarray())

with open('./pruebas/pruebas noticia/prueba 4 (IPN).txt', 'r', encoding='utf-8') as archivo:
    pruebatxt = archivo.read()

lematizarPrueba(pruebatxt)
"""
JEJEJEJJEJEJEEJJEJEJEJEJEJJEJEJEJEJE


def lematizarPrueba():
    
    #save the news sections in a file
    with open('corpus_noticias_secciones.txt', 'w', encoding='utf-8') as archivo:
        prueba = archivo.read()
        


    # Cargar el modelo de spaCy en español
    nlp = spacy.load("es_core_news_sm")

    # TOKENIZAR
    tokenized_prueba = []

    # Procesar el texto con SpaCy
    doc = nlp(prueba)

    # Obtener tokens del documento
    tokens = [token.text for token in doc]

    # Agregar la lista de tokens a 'tokenized_prueba'
    tokenized_prueba.append(tokens)

    # Imprimir la lista de tokens
    print(tokenized_prueba)
    # LEMATIZAR
    lemmatized_prueba = []

    # Convertir los tokens en un texto
    prueba_text = " ".join(tokens)

    # Procesar el texto con SpaCy para lematizar
    doc = nlp(prueba_text)

    # Obtener lemas de los tokens
    lemmas = [token.lemma_ for token in doc]

    # Agregar la lista de lemas a 'lemmatized_prueba'
    lemmatized_prueba.append(lemmas)

    # Imprimir la lista de lemas
    print(lemmatized_prueba)


    # Remove stop words that are articles,prepositions,conjunctions and pronouns en español
    # Agregar signos de puntuación a la lista de stopwords
    #punctuation = ['.', ',', ';', '!', '?', ':', '-', '"', '(', ')', '[', ']', '{', '}', '...', '¡', '¿', '»', '«', '...', '``', "''", '/', '|', '“', '”','*','él']

    stopwords  = [
        "el", "la", "los", "las", "un", "una", "unos", "unas","la","lo","a","al","del"  # Artículos
        "a", "ante", "bajo", "con", "contra", "de", "desde", "en", "entre", "hacia", "para", "por", "según", "sin", "sobre", "tras",  # Preposiciones
        "y", "o", "pero", "ni", "que", "si", "como", "porque","u",  # Conjunciones
        "yo", "tú", "él", "ella", "nosotros", "vosotros", "ellos", "ellas", "mí", "ti", "sí", "nos", "vos", "se", "me", "te", "le", "nos", "os", "les", "se",  # Pronombres personales
    ]

    #stopwords = stopwords.union(punctuation)#spacy.lang.es.stop_words.STOP_WORDS.union(punctuation)

    # ELIMINAR STOPWORDS
    filtered_tokens = [token for token in lemmatized_prueba if token.lower() not in stopwords]

    # Imprimir la lista de tokens lematizados sin stopwords
    print(filtered_tokens)

    # Save the filtered corpus in a file
    with open('corpus_final.txt', 'w', encoding='utf-8') as archivo:
        for section in filtered_tokens:
            section_text = " ".join(section)
            archivo.write(section_text + '\n')
"""
