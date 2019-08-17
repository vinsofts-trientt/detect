#dự đoán tuổi, giới tính, cảm xúc và vẽ nó trên face
import cv2 as cv
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

class Age_Gender_Emo:
    def __init__(self,ageProto,ageModel,genderProto,genderModel,emotion_path):
        # print(ageProto)
        self.ageNet = cv.dnn.readNet(ageModel, ageProto)
        self.genderNet = cv.dnn.readNet(genderModel, genderProto)
        self.model_emotion = load_model(emotion_path,compile = False)


    def detect_Age_Gender_Emo(self,frame,bbox,_frame):
        if len(bbox) > 0 :
            ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
            genderList = ['Male', 'Female']
            EMOTIONS = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]  #ghê tởm,sợ hãi,ngạc nhiên
            padding = 20   
            MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
            # print("listBox",frame.shape[0])
            face = frame[max(0,bbox[0][1]-padding-50):min(bbox[0][3]+padding+10,frame.shape[0]-1),max(0,bbox[0][0]-padding):min(bbox[0][2]+padding, frame.shape[1]-1)]
            # face = frame[bbox[0][1]:bbox[0][3],bbox[0][0]:bbox[0][2],:]
            # cv.imshow("face",face)
            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            #dự đoán giới tính
            self.genderNet.setInput(blob)
            genderPreds = self.genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            #dự đoán tuổi
            self.ageNet.setInput(blob)
            agePreds = self.ageNet.forward()
            # print("agePreds[0][agePreds[0].argmax()]",agePreds[0][agePreds[0].argmax()])
            if agePreds[0][agePreds[0].argmax()] < 0.5:
                age = "(25-32)"
            else:
                age = ageList[agePreds[0].argmax()]
            # print(age)
            # age = ageList[agePreds[0].argmax()]
            #dự đoán cảm xúc
            roi = cv.cvtColor(face, cv.COLOR_BGR2GRAY) #chuyển về thang màu xám
            roi = cv.resize(roi,(48,48))
            # cv.imshow("roi",roi)
            roi = roi.astype("float")/255.0  #convert sang float and  / 255
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            emotion = self.model_emotion.predict(roi)[0]
            # print("emotion",emotion)
            if emotion[emotion.argmax()] > 0.5:
                label_emotion = EMOTIONS[emotion.argmax()]
            else: label_emotion = "neutral"
            # print("emotion",emotion[emotion.argmax()])
            checkSad = ["scared","sad","angry"]
            checkNeutral = ["surprised","neutral","disgust"]
            if label_emotion in checkSad:
                label_emotion = "sad"
            elif label_emotion in checkNeutral:
                label_emotion = "neutral"

            label = "{},{},{}".format(gender, age,label_emotion)
            #puttext ở tọa độ gốc của face
            cv.putText(_frame, label, (bbox[0][0], bbox[0][1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA) 
            # cv.imshow("Age Gender Demo", frameFace)
            # cv.imwrite("age-gender-out-{}".format(args.input),frameFace)
        return _frame

