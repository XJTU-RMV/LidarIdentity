import cv2
import detect
cap=cv2.VideoCapture('1.mp4')
a=detect.detectapi(weights='weights/best.pt')
while True:

    rec,img = cap.read()

    result,names =a.detect([img])
    #img=result[0][0] #第一张图片的处理结果图片
    
    for cls,(x1,y1,x2,y2),conf in result[0][1]: #第一张图片的处理结果标签。
        print(cls,x1,y1,x2,y2,conf)
        if cls==0 and conf >=0.5:
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0))
            cv2.putText(img,names[cls],(x1,y1-20),cv2.FONT_HERSHEY_DUPLEX,1.5,(255,0,0))
    res_img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
    cv2.imshow("vedio",res_img)

    if cv2.waitKey(1)==ord('q'):
        break

