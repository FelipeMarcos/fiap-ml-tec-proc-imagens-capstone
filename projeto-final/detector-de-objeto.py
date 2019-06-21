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

def detector(imagem, template):
	# Conversão da imagem par escala de cinza
	img = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

	orb = cv2.ORB_create(1000, 1.2)
	kp1, des1 = orb.detectAndCompute(img, None)
	kp2, des2 = orb.detectAndCompute(template, None)

	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	matches = bf.match(des1, des2)
	matches = sorted(matches, key=lambda val: val.distance)
	return len(matches)

# Carregue a imagem do logotipo
image_template = cv2.imread("imagens/logo.png", 0)
cap = cv2.VideoCapture(0)

while True:
	# Obtendo imagem da câmera
	ret, frame = cap.read()
	
	if ret:
		# Definindo região de interesse (ROI)
		height, width = frame.shape[:2]
		top_left_x = int(width / 3)
		top_left_y = int((height / 2) + (height / 4))
		bottom_right_x = int((width / 3) * 2)
		bottom_right_y = int((height / 2) - (height / 4))
	
		# Desenhar retângulo na região de interesse
		cv2.rectangle(frame, (top_left_x,top_left_y), (bottom_right_x,bottom_right_y), 255, 3)

		# Obtendo região de interesse para validação do detector
		cropped = frame[bottom_right_y:top_left_y , top_left_x:bottom_right_x]

		# Executando o detector, definindo um limiar e fazendo a comparação para validar se o logotipo foi detectado
		matches = detector(cropped, image_template)

		if matches > 80:
			cv2.putText(frame, "logotipo encontrado", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 2)

		cv2.imshow("Identificacao de Template", frame)
		
	# Se for teclado Enter (tecla 13) deverá sair do loop e encerrar a captura de imagem
	if cv2.waitKey(1) == 13: 
		break

cap.release()
cv2.destroyAllWindows()

# IMPLEMENTAR
# Passe o parâmetro localização da imagem para exibi-la no notebook

# exibir_imagem(None)