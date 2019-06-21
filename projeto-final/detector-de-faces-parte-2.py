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

# Defina o diretório utilizado para salvar as faces de exemplo
data_path = 'faces/'

onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
training_data, labels = [], []

for i, files in enumerate(onlyfiles):
	image_path = data_path + onlyfiles[i]
	images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
	training_data.append(images)
	labels.append(0)

# Criando uma matriz da lista de labels
labels = np.asarray(labels, dtype=np.int32)

# Treinamento do modelo
model = cv2.face.LBPHFaceRecognizer_create()
model.train(training_data, labels)

print("Modelo treinado com sucesso.")

# IMPLEMENTAR
# Defina na chave 0 o nome do candidato

persons = {0: "Felipe"}

cap = cv2.VideoCapture(0)

while True:
	ret, frame = cap.read()

	# Extraia a face da imagem obtida da câmera
	face = face_extractor(frame)
	
	try:
		# Faça os ajustes necessários para classificá-la no classifcador treinado
		face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
		result = model.predict(face)

		# Estabeleça um algoritmo para concluir se o resultado é 'Sucesso', candidato identificado ou 'Não Indetificado' para quando não for localizado o candidato
		if result[1] < 500:
			confidence = int(100*(1-(result[1])/300))
			text = str(confidence) + "% de certeza"
			cv2.putText(frame, text, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

		if confidence > 75:
			text = persons[0] + " reconhecido"
			cv2.putText(frame, text, (100, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
			cv2.imshow('Imagem', frame)

		else:
			cv2.putText(frame, "Nao reconhecido", (100, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
			cv2.imshow('Imagem', frame)
	except:
		# Analise também situações onde a face não é identificada
		cv2.putText(frame, "Nao identificado", (100, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
		cv2.imshow('Imagem', frame)
		pass
	
		
	# Se for teclado Enter (tecla 13) deverá sair do loop e encerrar a captura de imagem    
	if cv2.waitKey(1) == 13: #13 is the Enter Key
		break
		
cap.release()
cv2.destroyAllWindows()     