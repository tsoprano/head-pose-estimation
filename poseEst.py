import cv2
import mediapipe as mp
import numpy as np
import time

#ML solution provided by mediapipe for detecting face mesh
mp_face_mesh = mp.solutions.face_mesh

#need confidence score higher than 0.5 to detect face 
#and mesh from frame to frame
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(0.5)


#setting up thickness and circle radius of the mesh 
#drawn on the detected face
mp_drawing = mp.solutions.drawing_utils
drwaing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# cap = cv2.VideoCapture("PilotData_DriverFatigue_GoPro.mp4")
cap = cv2.VideoCapture(0)


while cap.isOpened():
    success, image = cap.read()

    
    #to take time of how long the algorithm takes to work and to calculate
    #frames per second
    start = time.time()
    
    #the models we are paasing through mediapipe use RGB where openCV
    #reads in the images as RGB. Image is also flipped to maintain the
    #left and right head movement
    image = cv2.cvtColor(cv2.flip(image,1), cv2.COLOR_BGR2RGB)
    
    #we need the frames passed in the NN to be readable to improve performance
    image.flags.writeable = False
    
    #passing the frames to the face_mesh object created above, to get all the different landmarks in the face. 
    #First it runs as a face detector. It then crops the face to input it to the NN.
    results = face_mesh.process(image)

    results_mark = faceDetection.process(image)
    bboxs = []
    
    #changing iamge to writable again to draw on the image
    image.flags.writeable = True
    
    #converting the image back to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    
    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    #face detection
    if results_mark.detections:
	    for id, detection in enumerate(results_mark.detections):
	        bboxC = detection.location_data.relative_bounding_box
	        ih, iw, ic = image.shape
	        bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
	               int(bboxC.width * iw), int(bboxC.height * ih)
	        cx, cy = bbox[0] + (bbox[2] // 2), \
	                 bbox[1] + (bbox[3] // 2)
	        bboxInfo = {"id": id, "bbox": bbox, "score": detection.score, "center": (cx, cy)}
	        bboxs.append(bboxInfo)
	        image = cv2.rectangle(image, bbox, (255, 0, 255), 2)

	        cv2.putText(image, f'{int(detection.score[0] * 100)}%',
	        	(bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
	        	2, (255, 0, 255), 2)
    
    #pose detection
    if results.multi_face_landmarks:
        #if one person in the image, one result
        for face_landmark in results.multi_face_landmarks:
            
            #running through all the landmarks (nose, mouth ends, eye ends)
            for idx, lm in enumerate(face_landmark.landmark):

                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    #separating out nose coordinates, used to draw a line from the nose as a guide to where
                    #the person is seeing
                    if idx ==1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.x * 3000)
                    
                    #scaling the normalized x,y value
                    x,y = int(lm.x*img_w), int(lm.y*img_h)
                    
                    face_2d.append([x,y])
                    face_3d.append([x, y, lm.z])
                    
            face_2d = np.array(face_2d, dtype = np.float32)
            face_3d = np.array(face_3d, dtype = np.float32)
            
            
            #assumed intrinsic parameters of the camera
            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0.0, img_h/2],
                                  [0.0, focal_length, img_w/2],
                                  [0.0, 0.0, 1.0]], dtype=np.float32)
            
            #distrotion parameters, camera assumed to be not distorted
            dist_matrix = np.zeros((4,1), dtype=np.float32)
            
            #Solving PnP: 
            #rot_vec: how much is the point actually in the image
            #trans_vec: how much is the our point translated
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            
            #Getting Rotational matrix from rot_vec
            rmat, jac = cv2.Rodrigues(rot_vec)
            
            #Getting angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            
            #y rotation degrees to display on the image
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360
            
            # #for webcam: where the head is tilted
            # if y<-10:
            #     text = "Looking left"
            # elif y>10:
            #     text = "Looking right"
            # elif x<-10:
            #     text = "Looking down"
            # elif x>10:
            #     text = "Looking up"
            # else:
            #     text = "Forward"

            #for provided video: where the head is tilted
            if y>0:
                text = "Looking left"
            elif y<-8:
                text = "Looking right"
            elif x<-10:
                text = "Looking down"
            elif x>10:
                text = "Looking up"
            else:
                text = "Forward"
            
            #to display the nose direction (where the face would be seeing)
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            #nose tip coordinates
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            #scaled based how much is the face looking in either directions
            p2 = (int(nose_2d[0] +y*10), int(nose_2d[1] -x*10))

            cv2.line(image, p1, p2, (255,0,0) ,3)

            # Adding the text on the image
            cv2.putText(image, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "x: "+str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "y: "+str(np.round(y,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "z: "+str(np.round(z,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        end = time.time()
        totalTime = end - start
        fps = 1/totalTime
        print("FPS: ", fps)
        cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Head Pose Estimation', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()