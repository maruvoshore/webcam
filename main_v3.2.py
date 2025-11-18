import cv2

# 얼굴 + 스마일 검출 모델
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                    "haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     "haarcascade_smile.xml")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 검출
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:
        # 얼굴 박스
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)

        # 👄 얼굴의 아래쪽 1/2만 사용 (입 주변 위주)
        mouth_gray = gray[y + h // 2: y + h, x: x + w]

        # 스마일 검출 (조건 살짝 완화)
        smiles = smile_cascade.detectMultiScale(
            mouth_gray,
            scaleFactor=1.7,    # 너무 크면 안 잡힘, 너무 작으면 오탐
            minNeighbors=18,    # v3.1보다 낮춤
            minSize=(40, 40)    # 너무 작으면 노이즈, 너무 크면 못 잡음
        )

        if len(smiles) > 0:
            expression = "SMILE"
            color = (0, 255, 255)
        else:
            expression = "NEUTRAL"
            color = (255, 0, 0)

        cv2.putText(frame, expression, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Expression Detection (v3.2)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
