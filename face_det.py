import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
vd_cap = cv2.VideoCapture(0)

while True:

    success_cap, frame = vd_cap.read()
    if not success_cap:
        break

    cnvrt_to_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dt_faces = face_cascade.detectMultiScale(
        cnvrt_to_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in dt_faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vd_cap.release()
cv2.destroyAllWindows()