import cv2

# -------------------------------------------------------------
# 1. 얼굴 + 스마일 모델 로드
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

# -------------------------------------------------------------
# 3. 스마일 연속 감지 카운트 (오탐 방지)
# -------------------------------------------------------------
smile_counter = 0
SMILE_THRESHOLD = 4   # 연속 4프레임 이상 웃어야 SMILE로 판정

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # 거울 모드
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---------------------------------------------------------
    # 4. 얼굴 검출
    # ---------------------------------------------------------
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(120, 120)    # 더 큰 얼굴만 처리 (오탐 줄임)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)

        # 얼굴의 하단 절반만 사용 (입/턱 영역)
        roi_gray = gray[y + int(h * 0.45): y + h, x: x + w]

        # -----------------------------------------------------
        # 5. 스마일 검출 (조건 강화)
        # -----------------------------------------------------
        smiles = smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.8,       # 더 강한 축소 → 더 확실한 웃음만 잡힘
            minNeighbors=35,       # 더 높은 신뢰도
            minSize=(60, 60)       # 작은 움직임은 무시
        )

        # -----------------------------------------------------
        # 6. 스마일 연속 감지 로직
        # -----------------------------------------------------
        if len(smiles) > 0:
            smile_counter += 1
        else:
            smile_counter = max(0, smile_counter - 1)  # 감소(너무 급하지 않게)

        # 웃음 판정
        if smile_counter >= SMILE_THRESHOLD:
            expression = "SMILE"
            color = (0, 255, 255)
        else:
            expression = "NEUTRAL"
            color = (255, 0, 0)

        cv2.putText(frame, expression, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Expression Detection (v3.1)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()