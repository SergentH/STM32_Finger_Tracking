#Code Finger tracking pour STM32 sous opencv
# Sergent hugo
# Doktor Thibault
# 2019
from threading import Thread, Lock
import numpy as np
import cv2 as cv
import time

class WebcamVideoStream :
    def __init__(self, src = 0, width = 320, height = 240) :
        self.stream = cv.VideoCapture(src)
        self.stream.set(cv.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def start(self) :
        if self.started :
            print("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self) :
        while self.started :
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self) :
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def stop(self) :
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback) :
        self.stream.release()

if __name__ == "__main__" :
	#lancement du thread de recuperation des images
    vs = WebcamVideoStream().start()
    
    attente_camera = 0
    
    t0 = time.clock()
    actual_time = 0
    nb_loop = 0

	#initialisation du point au bout du doigt
    p0 = []
    
    #Les images pour la le détecteur
    frame_zero = 0
    frame_one = 0
    frame_two = 0
    frame_three = 0
    frame_four = 0
    
    
    first_frame_compare = 0
    Is_detected = 0
    passage = -1
    first_passage = 0
    hand_hist = 0

	#chargement des différentes images
    facture = cv.imread('fac.jpg')
    facture = cv.resize(facture, (320, 240)) 

    finger = cv.imread("finger.png")
    finger = cv.resize(finger, (320, 240))
    finger_up = cv.imread("finger_up.png")
    finger_up = cv.resize(finger_up, (320, 240)) 
    carres = cv.imread("carres.png")
    carres = cv.resize(carres, (320, 240)) 
    
    exemple = cv.imread("exemple.png")
    exemple = cv.resize(exemple, (320, 240)) 
    suite = cv.imread("A_vous.png")
    suite = cv.resize(suite, (320, 240)) 
    rectangle = cv.imread("rectangle.png")
    rectangle = cv.resize(rectangle, (320, 240)) 
    signature1 = cv.imread("signature1.jpg")
    signature1 = cv.resize(signature1, (320, 240))
    signature2 = cv.imread("signature2.jpg")
    signature2 = cv.resize(signature2, (320, 240))
    signature3 = cv.imread("signature3.jpg")
    signature3 = cv.resize(signature3, (320, 240))
    
	#definition de la fenetre
    window_name = 'frame'
    cv.namedWindow(window_name, cv.WND_PROP_FULLSCREEN)
    cv.moveWindow(window_name, 0, 0)
    cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN,
                          cv.WINDOW_FULLSCREEN)

    def contours(hist_mask_image):
        gray_hist_mask_image = cv.cvtColor(hist_mask_image, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray_hist_mask_image, 0, 255, 0)
        _, cont, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        return cont

    def centroid(max_contour):
        moment = cv.moments(max_contour)
        if moment['m00'] != 0:
            cx = int(moment['m10'] / moment['m00'])
            cy = int(moment['m01'] / moment['m00'])
            return cx, cy
        else:
            return None

    def farthest_point(defects, contour, centroid):
        if defects is not None and centroid is not None:
            s = defects[:, 0][:, 0]
            cx, cy = centroid	
            x = np.array(contour[s][:, 0][:, 0], dtype=np.float)
            y = np.array(contour[s][:, 0][:, 1], dtype=np.float)
           
            yIndice = np.argmin(y)
            xPoint = x[yIndice]
            yPoint = y[yIndice]

            farthest_point = [ int(xPoint) , int(yPoint)]
            return farthest_point

    
    while len(p0) == 0:
		#recuperation d'un image depuis le thread
        old_frame = vs.read()
        
        rows,cols,_ = old_frame.shape
        
		#partie pour afficher le tutoriel
        if passage == -1:
            attente_camera =  attente_camera +1
                
            old_frame = cv.add(old_frame,  exemple)
            if attente_camera < 75:
                old_frame = cv.add(old_frame,  carres)
                old_frame = cv.add(old_frame,  finger)
            if (attente_camera>=75) and (attente_camera < 125):
                old_frame = cv.add(old_frame,  carres)
                old_frame = cv.add(old_frame,  finger_up)
            if (attente_camera>=125) and (attente_camera < 200):
                old_frame = cv.add(old_frame,  rectangle)
            if (attente_camera>=200) and (attente_camera < 250):
                old_frame = cv.add(old_frame,  signature1)
            if (attente_camera>=250) and (attente_camera < 300):
                old_frame = cv.add(old_frame,  signature2)
            if (attente_camera>=300) and (attente_camera < 350):
                old_frame = cv.add(old_frame,  signature3)
            if (attente_camera>=350) and (attente_camera < 425):
                old_frame = cv.add(old_frame,  signature3)
                old_frame = cv.add(old_frame,  suite)
            
            if attente_camera >= 425:
                passage = 0
        
        #Creation des differents rectangles pour l'histogramme
        if passage == 0 : 
            hand_row_nw = np.array([60*rows/100,60*rows/100,60*rows/100,65*rows/100,65*rows/100,65*rows/100,70*rows/100,70*rows/100,70*rows/100])
            hand_col_nw = np.array([47*cols/100,50*cols/100,54*cols/100,47*cols/100,50*cols/100,54*cols/100,47*cols/100,50*cols/100,54*cols/100])

            hand_row_se = hand_row_nw + 10
            hand_col_se = hand_col_nw + 10

            size = hand_row_nw.size
            for i in range(size):
                cv.rectangle(old_frame,( int(hand_col_nw[i]), int(hand_row_nw[i]) ),( int(hand_col_se[i]),int(hand_row_se[i]) ),(0,255,0),1)

        #faire la différence entre la première image et celle en cours
        #Que la webcam s'aclimate et regle ses couleurs 
            if first_passage == 4:
                frame_four = old_frame
                first_passage = first_passage +1
            if first_passage == 3:
                frame_three = old_frame
                first_passage = first_passage +1
            if first_passage == 2:
                frame_two = old_frame
                first_passage = first_passage +1
            if first_passage == 1:
                frame_one = old_frame
                first_passage = first_passage +1
            if first_passage == 0:
                frame_zero = old_frame
                first_passage = first_passage +1
            
            if first_passage >= 5:
                
                #creation du rectangle declencheur
                y_haut = int(50*rows/100)
                x_haut = int(47*cols/100)
                y_bas = int(53*rows/100)
                x_bas = int(50*cols/100)
                cv.rectangle(old_frame,( int(x_haut)-1, int(y_haut)-1 ),(int(x_bas),int(y_bas) ),(255,0,255),1)
                #nouvelle image a comparer
                old_frame_compare = old_frame[y_haut:y_bas , x_haut: x_bas]
                
                #on prends il y a 5 images
                first_frame_compare = frame_zero
                first_frame_compare = first_frame_compare[y_haut:y_bas , x_haut: x_bas]
        
                frame_zero = frame_one
                frame_one = frame_two
                frame_two = frame_three
                frame_three = frame_four
                frame_four = old_frame
                
                Diff_image = cv.absdiff(first_frame_compare, old_frame_compare);
                ret,thresh_diff = cv.threshold(Diff_image,25,255,cv.THRESH_BINARY)
                #fait la moyenne entre deux images afin de voir si il y a eu un mouvement
                Is_detected = np.mean(thresh_diff)
          
        #calcul de l'histogramme de la pigmentation de la main si déclenchement
            if Is_detected > 80:
                print("GO !")
                hsv = cv.cvtColor(old_frame, cv.COLOR_BGR2HSV)
                roi = np.zeros([90,10,3], dtype=hsv.dtype)

                size = hand_row_nw.size
                for i in range(size):
                    roi[i*10:i*10+10,0:10] = hsv[  int(hand_row_nw[i]):int(hand_row_nw[i])+10,int(hand_col_nw[i]):int(hand_col_nw[i])+10    ]

                hand_hist = cv.calcHist([roi],[0, 1], None, [180, 256], [0, 180, 0, 256])
                cv.normalize(hand_hist, hand_hist, 0, 255, cv.NORM_MINMAX)
                passage = 1

    #Mise en place du masque afin de cacher les parties autres que le couleur de l'histrogramme
        if passage == 1:
            dst = cv.calcBackProject([hsv], [0,1], hand_hist, [0,180,0,256], 1)
            disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15,15))
            cv.filter2D(dst, -1, disc, dst)
            ret, old_thresh = cv.threshold(dst, 100, 255, 0)
            old_thresh = cv.merge((old_thresh,old_thresh, old_thresh))
            
            #Partie qui genere le premier point de la signature
            contour_list = contours(old_thresh)
            max_cont= max(contour_list, key = cv.contourArea)
            cnt_centroid = centroid(max_cont)
            y_cent = cnt_centroid [1]
            if max_cont is not None:
                old_gray = old_thresh
                hull = cv.convexHull(max_cont, returnPoints=False)
                defects = cv.convexityDefects(max_cont, hull)
                far_point = farthest_point(defects, max_cont, cnt_centroid)
                if far_point[1] < y_cent:
                    p0 = np.array([[[np.float32(far_point[0]),np.float32(far_point[1])]]])
                else:
                    passage = 0
        
        cv.imshow(window_name, old_frame)
        
        nb_loop = nb_loop +1
        
        #calcul des images par seconde
        if nb_loop == 10:
            actual_time = time.clock()
            calcul_fps = nb_loop / (actual_time - t0)
            print("nombre frame attente: ",  calcul_fps)
            t0 = actual_time
            nb_loop = 0
        
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (30,30),
                      maxLevel = 2,
                      criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = (0,0,255)

    Finished = 0
    ok_passage = 0
    facture_finale = np.zeros_like(facture)
    frame_rows = 0
    frame_cols = 0
    good_new = 0
    
    
    t0 = time.clock()
    nb_loop = 0
    
    while(Finished != 1):
        #recuperation d'une image depuis le thread
        frame = vs.read()
        
        rows,cols,_ = facture.shape    
        
        #definition d'une image de base
        if ok_passage == 0 :
            OK_frame = frame
            ok_passage = 1

		#creation du rectangle de signature
        y_haut = int(10*rows/100)
        x_haut = int(10*cols/100)
        y_bas = int(90*rows/100)
        x_bas = int(90*cols/100)

        cv.rectangle(facture,( int(x_haut)-1, int(y_haut)-1 ),( int(x_bas),int(y_bas) ),(0,0,255),1)
     
        #binarisation de l'image suivant l'histogramme
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv], [0,1], hand_hist, [0,180,0,256], 1)
        disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25,25))
        cv.filter2D(dst, -1, disc, dst)
        ret, thresh = cv.threshold(dst, 110, 255, 0)
        frame_gray = cv.merge((thresh,thresh, thresh))

        #calcul du point p1 etant celui se rapporchant le plus du point p0
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        if st == 1 :
            good_new = p1[st==1]
            good_old = p0[st==1]  
            # draw the tracks
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                
                #first point
                a,b = new.ravel()
                #second point
                c,d = old.ravel()
             
                facture = cv.line(facture,(a, b),(c,d), color, 2)
                facture_finale = cv.circle(facture,(a,b),2,color,-1)
                
        cv.imshow(window_name,facture_finale)
        
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        #Mise a jour de l'image précedente et du point précedent
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

        nb_loop = nb_loop +1
    
		#calcul des images par seconde
        if nb_loop == 10:
            actual_time = time.clock()
            calcul_fps = nb_loop / (actual_time - t0)
            print("nombre frame traitement: ",  calcul_fps)
            t0 = actual_time
            nb_loop = 0
            
        #Test de sortie de boucle
        pouty = p0[0][0][1]
        poutx = p0[0][0][0]
        if (poutx < x_haut) or (poutx > x_bas) or (pouty < y_haut) or (pouty > y_bas) :
            print("Finished")
            cv.imwrite("signature.jpg", facture_finale)
            Finished = 1
        
    vs.stop()
    cv.destroyAllWindows()
