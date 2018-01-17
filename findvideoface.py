import numpy as np
import cv2
import random
from multiprocessing import Process, Queue, Lock
import time
import dlib

def face_recog_dlib(framequeue,l,lfp,lfpname,fn):

    print("start...")

    detector = dlib.get_frontal_face_detector()    
    sp = dlib.shape_predictor("e:/shape_predictor_5_face_landmarks.dat")
    facerec = dlib.face_recognition_model_v1("e:/dlib_face_recognition_resnet_model_v1.dat")

    n = 0
    
    while(True):
        q_f,count = framequeue.get()
        n += 1
        print(str(count) + " " + str(n) + " (qsize: " + str(framequeue.qsize()))

        #gray = cv2.cvtColor(q_f, cv2.COLOR_BGR2GRAY)
        
        faces = detector(q_f, 1)

        if(len(faces)>0): 

            print(str(fn) + ": found person! Frame Count: " + str(count))

            # Draw a rectangle around the faces
            for i,r in enumerate(faces):

                # Get the landmarks/parts for the face in box d.
                shape = sp(q_f, r)
                face_descriptor = facerec.compute_face_descriptor(q_f, shape)

                distance = np.linalg.norm(np.array(face_descriptor) - np.array(lfp))

                print("DISTANCE: " + str(distance))

                if(distance<0.6):
                    cv2.rectangle(q_f, (r.left(),r.top()), (r.right(),r.bottom()), (0, 255, 0), 2)
                    cv2.putText(q_f, lfpname + "(" + str(distance) + ")", (r.left(), r.top()-10), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
                    print(str(fn) + ": found " + lfpname + "! Frame Count: " + str(count))
                    cv2.imwrite('mgh-f-' + str(count)+'.jpg',q_f)

    print("exit!")

def videoops(framequeue,l,fn):

    print("video job start...")
    
    cap = cv2.VideoCapture('f:/meigonghe.mp4')

    print(cap.get(5))
    #print cap.get(cv2.cv.CV_CAP_PROP_FPS)

    framestart = 24*60*65
    readall = False
    cap.set(cv2.CAP_PROP_POS_FRAMES,framestart)
    t=time.clock()
    framecount = 0

    while(cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()

        framecount += 1

        t=time.clock()-t
        f=framecount/t

        draw_str(frame, (20, 20), "frame count :  %f" % framecount)
        draw_str(frame, (20, 40), "queue length :  %f" % framequeue.qsize())
        draw_str(frame, (20, 60), "time use :  %f" % t)
        draw_str(frame, (20, 80), "frame rate :  %f" % f)

        framequeue.put((frame.copy(),framecount))    

        # Display the resulting frame
        cv2.imshow('frame',frame)

        ch = cv2.waitKey(1)
        if ch == 27:
            break

    readall = True

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    print("video exit!")

def draw_str(dst, target, s):
    x, y = target
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

def face_encode(q,d,s,f):
    face_descriptors = []

    faces = d(q, 1)
    for i,r in enumerate(faces):
        shape = s(q, r)
        face_descriptor = f.compute_face_descriptor(q, shape)
        face_descriptors.append(face_descriptor)

    return face_descriptors

if __name__ == '__main__': 

    # Create a lock object to synchronize resource access
    lock = Lock()
    
    framequeue = Queue(24*10)

    detector = dlib.get_frontal_face_detector()    
    sp = dlib.shape_predictor("e:/shape_predictor_5_face_landmarks.dat")
    facerec = dlib.face_recognition_model_v1("e:/dlib_face_recognition_resnet_model_v1.dat")

    #lookforpersons = []
    lfpname = "chenbaoguo"
    lfp = cv2.imread('e:/lfp1.jpg')
    #gray = cv2.cvtColor(lfp, cv2.COLOR_BGR2GRAY)
    lfp_encodes = face_encode(lfp,detector,sp,facerec)

    threadn = cv2.getNumberOfCPUs()    
    facejobs = []
    for i in range(8):
        p = Process(target=face_recog_dlib, args=(framequeue,lock,lfp_encodes[0],lfpname,i))
        p.daemon = True
        facejobs.append(p)
        p.start()

    videojobs = []
    for j in range(1):
        vp = Process(target=videoops, args=(framequeue,lock,j))
        videojobs.append(vp)
        vp.start()
        vp.join()

    print("main exit!")