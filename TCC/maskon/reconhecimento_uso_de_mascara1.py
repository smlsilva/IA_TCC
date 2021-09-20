import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def setImagem(img):

    classificador = cv.CascadeClassifier(cv.data.haarcascades +  "haarcascade_frontalface_default.xml")

    img = cv.imread(img)
    img_colorida = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    cinza = cv.cvtColor(img_colorida, cv.COLOR_RGB2GRAY)
    faces = classificador.detectMultiScale(cinza, 1.1, 5, minSize=(150,150))
    
    for x,y,w,h in faces:
        img = cv.rectangle(img_colorida, (x,y), (x+w, y+h), (0, 0, 255), 2)
    
    plt.imshow(img_colorida)
    plt.show()

setImagem("csdsca.jpg")
