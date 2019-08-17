#dự đoán tư thế đầu và tính toán số face thỏa mãn,vẽ box trên face
from imutils import face_utils
import imutils
import dlib
import cv2
import numpy as np

class Head_Pose:
    def __init__(self, path_shape_predictor):
        self.image_points = np.array([
                            (359, 391),     # Nose tip 34
                            (399, 561),     # Chin 9
                            (337, 297),     # Left eye left corner 37
                            (513, 301),     # Right eye right corne 46
                            (345, 465),     # Left Mouth corner 49
                            (453, 469)      # Right mouth corner 55
                        ], dtype="double")
        # 3D model points.
        self.model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip 34
                            (0.0, -330.0, -65.0),        # Chin 9
                            (-225.0, 170.0, -135.0),     # Left eye left corner 37
                            (225.0, 170.0, -135.0),      # Right eye right corne 46
                            (-150.0, -150.0, -125.0),    # Left Mouth corner 49
                            (150.0, -150.0, -125.0)      # Right mouth corner 55

                        ])

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(path_shape_predictor)

    def Detect_Head_Pose(self, frame):
        # frame = imutils.resize(frame, width=1024, height=576)
        # frame = imutils.resize(frame, width=500, height=300)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape
        # detect faces in the grayscale frame
        rects = self.detector(gray, 0)
        count = 0
        listBox = []

        # if len(rects) > 0:
        #         text = "{} face(s) found".format(len(rects))
        #         cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.5, (0, 0, 255), 2)
        
        #lặp trên các khuôn mặt phát hiện
        for rect in rects:
            # tính tọa độ giới hạn khuôn mặt và vẽ nó lên frame
            (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
            listBox.append([bX,bY,bX+bW,bY+bH])

            # tọa độ face (bX, bY) là tọa độ điểm trên cùng bên trái
            #và  (bX + bW, bY + bH) là tọa độ dưới cùng bên phải
            # print("tọa độ face", bX,bY,bX + bW, bY + bH)
            #vẽ hộp giới hạn
            cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH),
                    (0, 255, 0), 1)

            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)  #list tọa độ mốc mặt
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw each of them
            # print("shape",shape[14]) 
            # print("shape",shape[4])       
            for (i, (x, y)) in enumerate(shape):  #lặp trên 68 mốc mặt
                if i == 33:
                    #tính toán lại tọa độ image_points
                    self.image_points[0] = np.array([x,y],dtype='double')
                    # write on frame in Green
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                elif i == 8:
                    self.image_points[1] = np.array([x,y],dtype='double')
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                elif i == 36:
                    self.image_points[2] = np.array([x,y],dtype='double')
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                elif i == 45:
                    self.image_points[3] = np.array([x,y],dtype='double')
                    # write on frame in Green
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                elif i == 48:
                    self.image_points[4] = np.array([x,y],dtype='double')
                    # write on frame in Green
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                if i == 54:
                    self.image_points[5] = np.array([x,y],dtype='double')
                    # write on frame in Green
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                else:
                    #tất cả các mốc còn lại vẽ màu đỏ
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                    cv2.putText(frame, str(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

                focal_length = size[1]
                center = (size[1]/2, size[0]/2)
                camera_matrix = np.array([[focal_length,0,center[0]],[0, focal_length, center[1]],[0,0,1]], dtype="double")

                # print("Camera Matrix :\n {0}".format(camera_matrix))

                dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
                (success, rotation_vector, translation_vector) = cv2.solvePnP(self.model_points, self.image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)#flags=cv2.CV_ITERATIVE)

                # print("Rotation Vector:\n {0}".format(rotation_vector))
                # print("Translation Vector:\n {0}".format(translation_vector))

                # Project a 3D point (0, 0 , 1000.0) onto the image plane
                # We use this to draw a line sticking out of the nose_end_point2D
                # projectPoints hàm có tác dụng chiếu tọa độ 3d xuống 2d
                (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]),rotation_vector, translation_vector, camera_matrix, dist_coeffs)

            for p in self.image_points:
                    cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
            # print("nose_end_point2D",nose_end_point2D)
            # print("jacobian",jacobian)

            p1 = ( int(self.image_points[0][0]), int(self.image_points[0][1]))
            p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            print("p1",p1)
            print("shape[23]",shape[23])
            print("shape[9]",shape[9])
            far14 = abs(shape[14][0] - p1[0]) # tọa độ góc trái face
            far4 = abs(shape[4][0] - p1[0]) #tọa độ góc phải face
            far23 = abs(shape[23][1] - p1[1]) #tọa độ lông mày
            far9 = abs(shape[9][1] - p1[1]) # tọa độ cằm
            # print("far14", far14,far4)
            result_far_ngang = far14/far4
            result_far_doc = far23/far9
            #<x ,x càng nhỏ thì càng quay sang trái  , ngc lại
            if 0.6 < result_far_ngang < 1.5:  #if 0.3 < result_far < 3:   if 0.46 < result_far_ngang < 1.95: 
                # if 0.8 < result_far_doc < 1:  #x<  x càng lớn thì gương mặt càng cúi sâu
                    #<x, x càng nhỏ thì càng hướng mặt lên cao
                # print("result_far",result_far)
                count += 1
            cv2.line(frame, p1, p2, (255,0,0), 2)
            cv2.circle(frame, (1, 666), 1, (0, 0, 255), -1)
        # print("listBox",listBox)

        return frame,count,listBox       


