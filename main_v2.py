import cv2

# -------------------------------------------------------------
# 1. 얼굴 검출 모델 불러오기 (OpenCV 기본 제공)
#    이 파일은 OpenCV 안에 내장된 'Haar Cascade' 모델.
# -------------------------------------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                    "haarcascade_frontalface_default.xml")

# -------------------------------------------------------------
# 2. 웹캠 열기
# -------------------------------------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없어요 ㅠㅠ")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 거울 모드로 보기 (희나가 원하는 방식)
    frame = cv2.flip(frame, 1)

    # ---------------------------------------------------------
    # 3. 얼굴 검출은 '흑백 이미지'에서 더 잘 동작함
    # ---------------------------------------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---------------------------------------------------------
    # 4. 얼굴 위치 찾기
    #    scaleFactor  : 이미지 축소 비율(정확도 조정)
    #    minNeighbors : 박스를 얼마나 확실할 때 그릴지
    # ---------------------------------------------------------
    faces = face_cascade.detectMultiScale(
        gray,          # 검출할 이미지
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)  # 작은 얼굴 무시
    )

    # ---------------------------------------------------------
    # 5. 검출된 얼굴마다 박스 그리기
    #    faces에는 (x, y, w, h) 형태의 리스트가 들어감
    # ---------------------------------------------------------
    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,          # 그릴 이미지
            (x, y),         # 박스 왼쪽 위 좌표
            (x+w, y+h),     # 박스 오른쪽 아래 좌표
            (0, 255, 0),    # 박스 색 (초록)
            2               # 두께
        )

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
