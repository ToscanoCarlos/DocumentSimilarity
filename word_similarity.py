import pandas as pd
import math 


def cosine(v, w):
	val = sum(v[index] * w[index] for index in range(len(v)))
	sr_v = math.sqrt(sum(v_val**2 for v_val in v))
	sr_w = math.sqrt(sum(w_val**2 for w_val in w))
	res = val/(sr_v*sr_w)
	print (res)


                     #~ computación  biología  artes
#~ computadora          300       	100     	25
#~ internet             200        	50     		70
#~ célula                10       	250      	3
#~ pintura                2        	10    		280





data_words = {'computadora': [300, 100, 25], 'internet': [200, 50, 70], 'célula': [10, 250, 3], 'pintura': [2, 10, 280]}

print ('Word Similarity')
cosine(data_words['computadora'], data_words['internet'])
cosine(data_words['computadora'], data_words['célula'])
cosine(data_words['computadora'], data_words['pintura'])












