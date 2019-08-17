from head_pose_estimation import Head_Pose
from Age_Gender import Age_Gender_Emo
import cv2

import pafy
url = 'https://www.youtube.com/watch?v=kV3famkRaA4'
vPafy = pafy.new(url)
play = vPafy.getbest(preftype="mp4") 

# urll = "C:/Users/anlan/OneDrive/Desktop/frame_violence/tram_xang2/test/test181.jpg"
urlVideo = "test1.mp4"
path_Shape = "shape_predictor_68_face_landmarks.dat"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
emotion_path = "model_emotion.h5"

lstSecond = []
# frame = cv2.imread(urll)
cap =cv2.VideoCapture(play.url)
_Head_Pose = Head_Pose(path_Shape)
_Age_Gender_Emo = Age_Gender_Emo(ageProto,ageModel,genderProto,genderModel,emotion_path)
while True:
    ret,frame = cap.read()
    _frame,count,bbox = _Head_Pose.Detect_Head_Pose(frame)
    if len(lstSecond) == 0:
        lstSecond.append(count)
    elif lstSecond[len(lstSecond)-1] != count:
        lstSecond.append(count)
    print("lstSecond",lstSecond)
    numberSecond = lstSecond[0]
    for i in range(0,len(lstSecond)-1):
        if lstSecond[i] < lstSecond[i+1]:
            numberSecond = numberSecond + (lstSecond[i+1]-lstSecond[i])
    _framee = _Age_Gender_Emo.detect_Age_Gender_Emo(frame,bbox,_frame)

    print("numberSecond",numberSecond)
    cv2.imshow("as",_framee)
    cv2.waitKey(1)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

