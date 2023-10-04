import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math 

# Función para calcular la similitud del coseno entre dos vectores
def cosine(x, y):
    val = sum(x[index] * y[index] for index in range(len(x)))
    sr_x = math.sqrt(sum(x_val**2 for x_val in x))
    sr_y = math.sqrt(sum(y_val**2 for y_val in y))
    res = val / (sr_x * sr_y)
    return res

# Función para normalizar el texto
def normalizar_texto(texto):
    nlp = spacy.load("es_core_news_sm")
    doc = nlp(texto)
    tokens = [token.text for token in doc]

    lemmatized_tokens = []
    texto_procesado = " ".join(tokens)
    doc = nlp(texto_procesado)

    stop_words = [
        "el", "la", "los", "las", "un", "una", "unos", "unas", "al", "del", "lo", "este", "ese", "aquel", "estos", "esos", "aquellos", "este", "esta", "estas", "eso", "esa", "esas", "aquello", "alguno", "alguna", "algunos", "algunas",
        "a", "ante", "bajo", "cabe", "con", "contra", "de", "desde", "en", "entre", "hacia", "hasta", "para", "por", "según", "sin", "so", "sobre", "tras", "durante", "mediante", "excepto", "a través de", "conforme a", "encima de", "debajo de", "frente a", "dentro de",
        "y", "o", "pero", "ni", "que", "si", "como", "porque", "aunque", "mientras", "siempre que", "ya que", "pues", "a pesar de que", "además", "sin embargo", "así que", "por lo tanto", "por lo que", "tan pronto como", "a medida que", "tanto como", "no solo... sino también", "o bien", "bien... bien",
        "yo", "tú", "él", "ella", "nosotros", "vosotros", "ellos", "ellas", "usted", "nosotras", "me", "te", "le", "nos", "os", "les", "se", "mí", "ti", "sí", "conmigo", "contigo", "consigo", "mi", "tu", "su", "nuestro", "vuestro", "sus", "mío", "tuyo", "suyo", "nuestro", "vuestro", "suyo"]

    tokens_lematizados = [token.lemma_ for token in doc if token.text.lower() not in stop_words]
    texto_normalizado = ' '.join(tokens_lematizados)

    with open('prueba_normalizada.txt', 'w', encoding='utf-8') as archivo:
        archivo.write(texto_normalizado)

    return texto_normalizado

# Leer noticias normalizadas desde un archivo
noticias = []
with open('corpus_normalizado.txt', 'r', encoding='utf-8') as archivo:
    lineas = archivo.readlines()
    for linea in lineas:
        noticias.append(linea.strip('\n'))

# Vectorización binaria
vectorizador_binario = CountVectorizer(binary=True)
xBinario = vectorizador_binario.fit_transform(noticias)
xArregloBinario = xBinario.toarray()

# Vectorización de frecuencia
vectorizador_frecuencia = CountVectorizer(token_pattern=r'(?u)\w\w+|\w\w+\n|\.')
xFrecuencia = vectorizador_frecuencia.fit_transform(noticias)
xArregloFrecuencia = xFrecuencia.toarray()

# Vectorización TF-IDF
vectorizador_tfidf = TfidfVectorizer(token_pattern=r'(?u)\w\w+|\w\w+\n|\.')
xtfidf = vectorizador_tfidf.fit_transform(noticias)
xArregloTfidf = xtfidf.toarray()

pruebaNoticia = []

# Función para cargar una noticia desde un archivo
def cargar_noticia():
    global pruebaNoticia
    ruta_archivo = filedialog.askopenfilename(filetypes=[('Archivos de texto', '*.txt')])
    if ruta_archivo:
        with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
            contenido = archivo.read()
        messagebox.showinfo('Información', 'Archivo de noticias cargado exitosamente.')
        pruebaNoticia = [normalizar_texto(contenido)]

# Función para mostrar las 10 noticias más similares
def mostrar_top_10(tipo_vectorizacion):
    global xArregloBinario, xArregloFrecuencia, xArregloTfidf, pruebaNoticia, vectorizador_binario, vectorizador_frecuencia, vectorizador_tfidf

    if tipo_vectorizacion == "Binaria":
        yArreglo = vectorizador_binario.transform(pruebaNoticia).toarray()
        xArreglo = xArregloBinario
    elif tipo_vectorizacion == "Frecuencia":
        yArreglo = vectorizador_frecuencia.transform(pruebaNoticia).toarray()
        xArreglo = xArregloFrecuencia
    else:
        yArreglo = vectorizador_tfidf.transform(pruebaNoticia).toarray()
        xArreglo = xArregloTfidf

    # Calcular la similitud del coseno entre la noticia de prueba y las noticias del corpus
    similitud = []
    for i in range(len(xArreglo)):
        similitud.append(cosine_similarity([xArreglo[i]], yArreglo)[0][0])

    # Encontrar las 10 noticias más similares
    top_10_noticias = []
    for i in range(10):
        indice = similitud.index(max(similitud))
        top_10_noticias.append((i + 1, indice, similitud[indice]))
        similitud[indice] = 0

    # Crear un texto con las noticias más similares y mostrarlo en la ventana
    texto_top_10 = "\n".join([f"{i}. Noticia: {indice + 1}, Similitud: {similitud:.4f}" for i, indice, similitud in top_10_noticias])
    resultado.config(state="normal")
    resultado.delete("1.0", "end")
    resultado.insert("1.0", "Top 10 noticias más similares:\n" + texto_top_10)
    resultado.config(state="disabled")

ventana = tk.Tk()
ventana.title("Buscador de Noticias")

cargar_btn = tk.Button(ventana, text="Cargar Noticia", command=cargar_noticia)
cargar_btn.pack()

etiqueta_vectorizacion = tk.Label(ventana, text="Seleccione el tipo de vectorización:")
etiqueta_vectorizacion.pack()

variable_vectorizacion = tk.StringVar(ventana)
variable_vectorizacion.set("Frecuencia")

opciones_vectorizacion = tk.OptionMenu(ventana, variable_vectorizacion, "Binaria", "Frecuencia", "TF-IDF")
opciones_vectorizacion.pack()

buscar_btn = tk.Button(ventana, text="Buscar Noticias Similares", command=lambda: mostrar_top_10(variable_vectorizacion.get()))
buscar_btn.pack()

resultado = tk.Text(ventana, height=10, width=60, state="disabled")
resultado.pack()

ventana.mainloop()
