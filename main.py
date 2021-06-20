import cv2
import numpy as np
import utlis

webCamFeed = False
pathImage = "2.jpg" # jika menggunakan gambar
cap = cv2.VideoCapture(0) # jika menggunakkan kamera
cap.set(10,160)
heightImg = 700
widthImg  = 700
questions=5
choices=5
ans= [0,2,0,2,3]

count=0
while True:
    # cek jika camera ada maka dapatkan gambar dari kamera
    if webCamFeed:success, img = cap.read()
    else:img = cv2.imread(pathImage) # jika tidak dapatkan dari gambar statis
    img = cv2.resize(img, (widthImg, heightImg)) # ubah ukuran gambar
    imgFinal = img.copy() #buatt duplikat gambar asli
    imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8) # buat gambar blank untuk debuugging
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # ubah gambar jadi grayscale
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) # tambahan gaussian blur pada gambar
    imgCanny = cv2.Canny(imgBlur,10,70) # tambahkan filter Canny pada gambar untuk mendapatkan garis tepi

    try:
        #GET ALL COUNTOURS
        imgContours = img.copy()
        imgBigContour = img.copy()
        contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # mendapatan semua garis tepi dengan fillter canny
        cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) # Contours yang di dapat di gambar pada gambar
        rectCon = utlis.rectContour(contours) # dapatkan contours yang membentuk kotak
        biggestPoints= utlis.getCornerPoints(rectCon[0]) # dapatkan titik sudut(kordinat) dari kotak terbesar
        gradePoints = utlis.getCornerPoints(rectCon[1]) # dapatkan titik sudut(kordinat) dari kotak terbesar ke 2

        if biggestPoints.size != 0 and gradePoints.size != 0: #di cek titik sudut di dapatkan
            # BIGGEST RECTANGLE WARPING (kotak jawaban)
            biggestPoints=utlis.reorder(biggestPoints) # urutkan titik koordinat agar saat di wrap sesuai urutan
            cv2.drawContours(imgBigContour, biggestPoints, -1, (0, 255, 0), 20) # gambar contour untuk kotak lembar jawaban
            pts1 = np.float32(biggestPoints) # menyiapkan titik sudut yang akan di lakukan wrap
            pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # mendefinisikan titik warp baru
            matrix = cv2.getPerspectiveTransform(pts1, pts2) # dapatkan transformation matrixnya
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg)) # dilakukan warping sesuai dengan transformation matrixnya

            # SECOND BIGGEST RECTANGLE WARPING (kotak nilai)
            cv2.drawContours(imgBigContour, gradePoints, -1, (255, 0, 0), 20)
            gradePoints = utlis.reorder(gradePoints)
            ptsG1 = np.float32(gradePoints)
            ptsG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
            matrixG = cv2.getPerspectiveTransform(ptsG1, ptsG2)
            imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))

            # APPLY THRESHOLD
            imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY) # gambar hasil wrap di jadikan gray
            imgThresh = cv2.threshold(imgWarpGray, 170, 255,cv2.THRESH_BINARY_INV )[1] # di beri threshold dan di invers

            boxes = utlis.splitBoxes(imgThresh) # gambar di split pe bulatan jawaban
            # cv2.imshow("Split Test ", boxes[3])
            countR=0
            countC=0
            myPixelVal = np.zeros((questions,choices)) # array untuk menyimpan nilai white pada setiap gambar bulatan jawaban
            for image in boxes:
                #cv2.imshow(str(countR)+str(countC),image)
                totalPixels = cv2.countNonZero(image) # mendapatkan total pixel yg tidak hitam pada gambar bulatan jawaban
                myPixelVal[countR][countC]= totalPixels # disimpan pada array
                countC += 1
                if (countC==choices):countC=0;countR +=1 # ganti baris

            # FIND THE USER ANSWERS AND PUT THEM IN A LIST
            myIndex=[] # untuk menyimpan jawaban user
            for x in range (0,questions): # di looping ber baris
                arr = myPixelVal[x] # dapatkan data per baris
                myIndexVal = np.where(arr == np.amax(arr)) # dapatkan index dengan nilai tertinggi
                myIndex.append(myIndexVal[0][0]) # simpan index pada list sebagai list dari jawaban pengisi
            #print("USER ANSWERS",myIndex)

            # COMPARE THE VALUES TO FIND THE CORRECT ANSWERS
            grading=[] # menyimpan nilai
            for x in range(0,questions):
                if ans[x] == myIndex[x]: # di cek jika jawaban sama dengan kunci jawaban
                    grading.append(1)
                else:grading.append(0)
            #print("GRADING",grading)
            score = (sum(grading)/questions)*100 # FINAL GRADE
            #print("SCORE",score)

            # DISPLAYING ANSWERS
            utlis.showAnswers(imgWarpColored,myIndex,grading,ans) # gambar indikator jawaban pada lembar jawaban yg sudah di wrap
            # imgGrid = utlis.drawGrid(imgWarpColored) # DRAW GRID
            imgRawDrawings = np.zeros_like(imgWarpColored) # buat gambar dengan ukuran yang sama dengan gambar hasil wrap
            utlis.showAnswers(imgRawDrawings, myIndex, grading, ans) # gambar indikator jawaban pada gambar yang baru di buat
            invMatrix = cv2.getPerspectiveTransform(pts2, pts1) # lalu dapatkan inverse matrix
            imgInvWarp = cv2.warpPerspective(imgRawDrawings, invMatrix, (widthImg, heightImg)) # gambar di invers dari wrap ke perspective semula

            # DISPLAY GRADE
            imgRawGrade = np.zeros_like(imgGradeDisplay,np.uint8) # buat gambar kosong dengan ukuran sama dengan kotak grade
            cv2.putText(imgRawGrade,str(int(score))+"%",(70,100)
                        ,cv2.FONT_HERSHEY_COMPLEX,3,(0,255,255),3) # tambahkan nilai pada gambar baru
            invMatrixG = cv2.getPerspectiveTransform(ptsG2, ptsG1) # dapatkan matrix inversenya
            imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg)) # gambar di invers dari wrap ke perspective semula

            # SHOW ANSWERS AND GRADE ON FINAL IMAGE
            #gabungkan hasil dari kotak jawaban yg sudah di beri indikator dan kotak grade yang sudah di beri nilai pada satu gambar utuh
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1,0)
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1,0)

            # IMAGE ARRAY FOR DISPLAY
            # menampilkan banyak gambar sekaligus dalam satu frame yang sama
            imageArray = ([img,imgGray,imgCanny,imgContours],
                          [imgBigContour,imgThresh,imgWarpColored,imgGrid])
            cv2.imshow("Final Result", imgFinal)
    except:
        imageArray = ([img,imgGray,imgCanny,imgContours],
                      [imgBlank, imgBlank, imgBlank, imgBlank])

    # LABELS FOR DISPLAY
    lables = [["Original","Gray","Edges","Contours"],
              ["Biggest Contour","Threshold","Warpped","Final"]]

    stackedImage = utlis.stackImages(imageArray,0.5,lables) # memberikan label untuk masing masing frame pada gambar
    cv2.imshow("Result",stackedImage)

    # SAVE IMAGE WHEN 's' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'): # jika user menekan tombol S
        # simpan gambarr
        cv2.imwrite("Scanned/myImage"+str(count)+".jpg",imgFinal)
        cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
                      (1100, 350), (0, 255, 0), cv2.FILLED)
        cv2.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow('Result', stackedImage)
        cv2.waitKey(300)
        count += 1