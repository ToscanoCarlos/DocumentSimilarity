import spacy
import re

archivo = "corpus_noticias.txt"
# archivo = "corpus_prueba.txt"
nlp = spacy.load("es_core_news_sm")

stop_words = [
"el", "la", "los", "las", "un", "una", "unos", "unas", "al", "del", "lo", "este", "ese", "aquel", "estos", "esos", "aquellos", "este", "esta", "estas", "eso", "esa", "esas", "aquello", "alguno", "alguna", "algunos", "algunas",
"a", "ante", "bajo", "cabe", "con", "contra", "de", "desde", "en", "entre", "hacia", "hasta", "para", "por", "según", "sin", "so", "sobre", "tras", "durante", "mediante", "excepto", "a través de", "conforme a", "encima de", "debajo de", "frente a", "dentro de",
"y", "o", "pero", "ni", "que", "si", "como", "porque", "aunque", "mientras", "siempre que", "ya que", "pues", "a pesar de que", "además", "sin embargo", "así que", "por lo tanto", "por lo que", "tan pronto como", "a medida que", "tanto como", "no solo... sino también", "o bien", "bien... bien",
"yo", "tú", "él", "ella", "nosotros", "vosotros", "ellos", "ellas", "usted", "nosotras", "me", "te", "le", "nos", "os", "les", "se", "mí", "ti", "sí", "conmigo", "contigo", "consigo", "mi", "tu", "su", "nuestro", "vuestro", "sus", "mío", "tuyo", "suyo", "nuestro", "vuestro", "suyo"]

archivo_salida = "corpus_normalizado_final.txt"

with open(archivo_salida, 'w', encoding='utf-8') as archivo_resultado: 
    with open(archivo, 'r', encoding='utf-8') as archivo_entrada:
        for linea in archivo_entrada:
            partes = linea.split("&&&&&&&&")
            texto_noticia = partes[2]
            doc = nlp(texto_noticia)
            tokens_lematizados = [token.lemma_ for token in doc if (
                token.text.lower() not in stop_words
                # token.text not in {"&", "|", ";", ".", "(", ")", "“", '"', ",", "*", "”", "!", ":", "-", "¿", "?", "[", "]", "‘", "’", "…", "«", "»",  "#", "/", "\\","¡", "—", "...", "_" , "$", "@"}
            )]
            noticia_normalizada = ' '.join(tokens_lematizados)
            archivo_resultado.write(f'{noticia_normalizada}\n')
        
print("Proceso de Normalizacion Terminado")


