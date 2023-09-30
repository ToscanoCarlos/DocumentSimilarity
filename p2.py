#Extract from the news corpus the news section and generate a new corpus with this content
#-Tokenize the extracted content and create a tokenized corpus
#-Lemmatize the corpus
#-Remove stop words that are articles,prepositions,conjunctions and pronouns
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
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
    return texto

"""
************************************************************************************
************************************************************************************
************************************************************************************
"""

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
xArregloBinario = xBinario.toarray()

vectorizador_frecuencia = CountVectorizer(token_pattern= r'(?u)\w\w+|\w\w+\n|\.')
xFrecuencia = vectorizador_frecuencia.fit_transform(noticias)
xArregloFrecuencia = xFrecuencia.toarray()

vectorizador_tfidf = TfidfVectorizer(token_pattern= r'(?u)\w\w+|\w\w+\n|\.')
xtfidf = vectorizador_tfidf.fit_transform(noticias)
xArregloTfidf = xtfidf.toarray()

pruebaNoticia = []

# Interfaz gráfica
def cargar_noticias():
    global pruebaNoticia
    archivo_path = filedialog.askopenfilename(filetypes=[('Archivos de texto', '*.txt')])
    if archivo_path:
        with open(archivo_path, 'r', encoding='utf-8') as archivo:
            pruebatxt = archivo.read()
        messagebox.showinfo('Información', 'Archivo de noticias cargado exitosamente.')
        pruebaNoticia = [lematizarPrueba(pruebatxt)]
        

def mostrar_top_10(tipo_vectorizacion):
    global  xArregloBinario, xArregloFrecuencia, xArregloTfidf, pruebaNoticia, vectorizador_binario, vectorizador_frecuencia, vectorizador_tfidf
    
    if tipo_vectorizacion == "Binaria":
        yArreglo = vectorizador_binario.transform(pruebaNoticia).toarray()
        xArreglo = xArregloBinario
    elif tipo_vectorizacion == "Frecuencia":
        yArreglo = vectorizador_frecuencia.transform(pruebaNoticia).toarray()
        xArreglo = xArregloFrecuencia
    else:
        yArreglo = vectorizador_tfidf.transform(pruebaNoticia).toarray()
        xArreglo = xArregloTfidf

    similitud = []
    for i in range(len(xArreglo)):
        similitud.append(cosine_similarity([xArreglo[i]], yArreglo)[0][0])

    top_10_noticias = []
    for i in range(10):
        index = similitud.index(max(similitud))
        top_10_noticias.append((i + 1, index, similitud[index]))
        similitud[index] = 0

    top_10_text = "\n".join([f"{i}. Noticia: {index}, Similitud: {similitud:.4f}" for i, index, similitud in top_10_noticias])
    resultado.config(state="normal")
    resultado.delete("1.0", "end")
    resultado.insert("1.0", "Top 10 noticias más similares:\n" + top_10_text)
    resultado.config(state="disabled")

ventana = tk.Tk()
ventana.title("Buscador de Noticias")

cargar_btn = tk.Button(ventana, text="Cargar Noticia", command=cargar_noticias)
cargar_btn.pack()

vectorizacion_label = tk.Label(ventana, text="Seleccione el tipo de vectorización:")
vectorizacion_label.pack()

vectorizacion_var = tk.StringVar(ventana)
vectorizacion_var.set("Frecuencia")

vectorizacion_opciones = tk.OptionMenu(ventana, vectorizacion_var, "Binaria", "Frecuencia", "TF-IDF")
vectorizacion_opciones.pack()

buscar_btn = tk.Button(ventana, text="Buscar Noticias Similares", command=lambda: mostrar_top_10(vectorizacion_var.get()))
buscar_btn.pack()

resultado = tk.Text(ventana, height=10, width=60, state="disabled")
resultado.pack()

ventana.mainloop()










