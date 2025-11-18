import cv2

# -------------------------------------------------------------
# 1. 얼굴(Frontal Face) + 스마일(Smile) 검출 모델 불러오기
# -------------------------------------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                    "haarcascade_frontalface_default.xml")

smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     "haarcascade_smile.xml")

# -------------------------------------------------------------
# 2. 웹캠 열기
# -------------------------------------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame error")
        break

    # 거울 모드(사용자 관찰용)
    frame = cv2.flip(frame, 1)

    # 얼굴/스마일 검출은 흑백에서 더 정확함
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---------------------------------------------------------
    # 3. 얼굴 검출
    # ---------------------------------------------------------
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100)
    )

    # ---------------------------------------------------------
    # 4. 얼굴마다 표정 분석 시작
    # ---------------------------------------------------------
    for (x, y, w, h) in faces:
        # 얼굴 박스 그리기
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)

        # 얼굴 영역 잘라내기 (ROI)
        face_gray = gray[y:y + h, x:x + w]
        face_color = frame[y:y + h, x:x + w]

        # -----------------------------------------------------
        # 5. 스마일 검출 (입꼬리 올라간 형태 찾기)
        # -----------------------------------------------------
        smiles = smile_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.7,
            minNeighbors=25,
            minSize=(40, 40)
        )

        # -----------------------------------------------------
        # 6. 표정 분류 (SMILE / NEUTRAL)
        # -----------------------------------------------------
        if len(smiles) > 0:
            expression = "SMILE"
            color = (0, 255, 255)   # 노란색
        else:
            expression = "NEUTRAL"
            color = (255, 0, 0)     # 파란색

        # 표정 텍스트 표시
        cv2.putText(frame, expression, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # ---------------------------------------------------------
    # 7. 프레임 화면 출력
    # ---------------------------------------------------------
    cv2.imshow("Expression Detection (v3)", frame)

    # q 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------------------------------------
# 8. 자원 해제
# -------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()