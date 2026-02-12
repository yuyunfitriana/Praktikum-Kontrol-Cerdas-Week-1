import cv2
import mediapipe
capture = cv2.VideoCapture(0)
mediapipehand = mediapipe.solutions.hands
tangan = mediapipehand.Hands(max_num_hands=2)
mpdraw = mediapipe.solutions.drawing_utils
while True:
    success, img = capture.read() #read video frame by frame
    image=cv2.flip(img,1)
    if not success:
        print("Webcam tidak terbaca")
        break
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = tangan.process(imgRGB)
    if results.multi_hand_landmarks:
        for titiktangan in results.multi_hand_landmarks:
            mpdraw.draw_landmarks( img,titiktangan,mediapipehand.HAND_CONNECTIONS)
            for id, titik in enumerate(titiktangan.landmark):
                print(id, titik.x, titik.y)
    if results.multi_handedness:
        for hand in results.multi_handedness:
            if hand.classification[0].index == 1:
                cv2.putText(img, "KIRI", (300, 450),
                            cv2.FONT_HERSHEY_PLAIN, 3, (50, 0, 500), 3)
            else:
                cv2.putText(img, "KANAN", (250, 450),
                            cv2.FONT_HERSHEY_PLAIN, 3, (50, 0, 500), 3)
    cv2.imshow("webcam", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break #TUTUP webcam dan jendela saat q tampilan ditekan
capture.release()
cv2.destroyAllWindows()
