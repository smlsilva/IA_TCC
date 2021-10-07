import cv2 as cv
import config
import os

cam = cv.VideoCapture(0)
classifier = cv.CascadeClassifier(f"{cv.haarcascades}{os.sep}haarcascade_frontalface_alt2.xml")

dataframe = config.load_dataframe()
X_train, X_test, y_train, y_test = config.train_test(dataframe)

pca = config.pca_model(X_train) # Modelo PCA para extração de features da imagem

X_train = pca.transform(X_train) # Conjunto de treino com features extraídas
X_test = pca.transform(X_test) # Conjunto de teste com features extraídas

knn = config.knn(X_train, y_train) #Treinando modelo classificatório KNN

label = {
    0: "Sem mascara",
    1: "Com mascara"
}

while True:
    status, frame = cam.read()

    if not status:
        break

    if cv.waitKey(1) & 0xff == ord('q'):
        break
    
    #Transformando a imagem em escala de cinza
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    #Detectando faces na imagem
    faces = classifier.detectMultiScale(gray)

    #Iterando nas faces encontradas
    for x,y,w,h in faces:
        gray_face = gray[y:y+h, x:x+w] #Recortand região da face

        if gray_face.shape[0] >= 200 and gray_face.shape[1] >= 200:
            gray_face = cv.resize(gray_face, (160,160)) #Redimensionando
            vector = pca.transform([gray_face.flatten()]) #Extraindo features da imagem

            pred = knn.predict(vector)[0] #Classificando a imagem
            classification = label[pred]

            #Desenhando retangulos em torno da face
            if pred == 0:
                cv.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 3)
                print("\a")
            elif pred == 1:
                cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
            
            #Escrevendo classificação e quantidade de faces vistas
            cv.putText(frame, classification, (x - 20,y + h + 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2, cv.LINE_AA)
            cv.putText(frame, f"{len(faces)} rostos identificados",(20,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2, cv.LINE_AA)

    #Mostrando o frame
    cv.imshow("Cam", frame)
