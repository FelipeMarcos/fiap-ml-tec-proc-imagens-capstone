#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
from scipy.spatial import distance as dist
import collections
from matplotlib.pyplot import figure

face_classifier = cv2.CascadeClassifier('classificadores/haarcascade_frontalface_default.xml')

def face_extractor(img):
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces = face_classifier.detectMultiScale(gray, 1.2, 5)
	
	if faces is ():
		return None
	
	for (x,y,w,h) in faces:
		cropped_face = img[y:y+h, x:x+w]

	return cropped_face

cap = cv2.VideoCapture(0)
contagem = 0
contagem_maxima = 100

while True:
	ret, frame = cap.read()

	if ret:
		cv2.imshow("Imagem de Treino", frame)
		
		if face_extractor(frame) is not None:
	        
			# Crie um algoritmo para salvar as imagens segmentadas em face em um determinado diretório
			contagem += 1

			face = cv2.resize(face_extractor(frame), (200, 200))
			face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

			file_name = 'faces/' + str(contagem) + '.jpg'
			cv2.imwrite(file_name, face)
			
			# Se for teclado Enter (tecla 13) deverá sair do loop e encerrar a captura de imagem
			# ou for alcançado a contagem máxima (amostras)
			if cv2.waitKey(1) == 13 or contagem == contagem_maxima:
				break
		
cap.release()
cv2.destroyAllWindows()
print("Coleta de amostras completado")