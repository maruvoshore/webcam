import cv2
import mediapipe as mp
import math

# -------------------------------------------------------------
# 1. MediaPipe 설정 (Face Mesh 사용)
# -------------------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,      # 눈/입 주변 더 정밀하게
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------------------------------------------------------------
# 2. 웹캠 열기
# -------------------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()


def get_distance(p1, p2):
    """두 점 사이 거리 계산 (유클리드 거리)"""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def classify_expression(landmarks, img_w, img_h):
    """
    입 랜드마크를 이용해서 SMILE / NEUTRAL 분류
    - mouth_ratio: 입이 얼마나 벌어졌는지 (치아 보이는 정도 근사)
    - corner_lift: 입꼬리가 중앙보다 얼마나 위로 올라갔는지
    """

    # Mediapipe FaceMesh의 입 주변 주요 인덱스
    LEFT_MOUTH = 61    # 왼쪽 입꼬리
    RIGHT_MOUTH = 291  # 오른쪽 입꼬리
    UPPER_LIP = 13     # 윗입술 중앙
    LOWER_LIP = 14     # 아랫입술 중앙

    lm = landmarks

    def to_pixel(idx):
        point = lm[idx]
        return int(point.x * img_w), int(point.y * img_h)

    lx, ly = to_pixel(LEFT_MOUTH)
    rx, ry = to_pixel(RIGHT_MOUTH)
    ux, uy = to_pixel(UPPER_LIP)
    bx, by = to_pixel(LOWER_LIP)

    mouth_width = get_distance((lx, ly), (rx, ry))
    mouth_height = get_distance((ux, uy), (bx, by))

    if mouth_width == 0:
        return "NEUTRAL"

    # 입이 얼마나 벌어졌는지 (세로/가로 비율)
    mouth_ratio = mouth_height / mouth_width

    # 입술 중앙 y좌표 (윗입술과 아랫입술 사이)
    center_y = (uy + by) / 2
    # 입꼬리 두 점의 평균 y좌표
    corner_y = (ly + ry) / 2

    # 입꼬리가 중앙보다 얼마나 위에 있는지 (위로 갈수록 y가 작으니까 부호 반대로)
    # 값이 클수록 입꼬리가 더 위로 올라간 상태라고 볼 수 있음
    corner_lift = (center_y - corner_y) / mouth_width  # 입 크기로 정규화

    # -----------------------------
    # 튜닝 가능한 기준값들
    # -----------------------------
    BIG_SMILE_RATIO = 0.18      # 크게 웃는 웃음 (입 많이 벌어진 상태)
    SMALL_SMILE_RATIO = 0.10    # 입 다문 미소(살짝 벌어진 상태)
    NEUTRAL_MOUTH_RATIO = 0.07  # 거의 안 벌린 입

    # 입꼬리 리프트 기준 (값이 클수록 더 위)
    STRONG_CORNER_LIFT = 0.03   # 확실히 웃는 형태
    WEAK_CORNER_LIFT = 0.01     # 살짝 위로 간 정도

    # 1) 크게 웃는 웃음: 입이 많이 벌어지고, 입꼬리도 확실히 위로
    if mouth_ratio > BIG_SMILE_RATIO and corner_lift > WEAK_CORNER_LIFT:
        return "SMILE"

    # 2) 입 다문 미소: 입은 조금만 벌어졌지만, 입꼬리가 꽤 위로간 상태
    if SMALL_SMILE_RATIO < mouth_ratio <= BIG_SMILE_RATIO and corner_lift > STRONG_CORNER_LIFT:
        return "SMILE"

    # 3) 거의 안 벌려진 입 + 입꼬리가 평평/아래 → NEUTRAL
    if mouth_ratio < NEUTRAL_MOUTH_RATIO and corner_lift < WEAK_CORNER_LIFT:
        return "NEUTRAL"

    # 4) 애매한 구간은 일단 NEUTRAL
    return "NEUTRAL"


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # 거울 모드
    img_h, img_w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    expression = "NEUTRAL"
    color = (255, 0, 0)  # NEUTRAL: 파랑

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # 표정 분류
            expression = classify_expression(landmarks, img_w, img_h)

            if expression == "SMILE":
                color = (0, 255, 255)  # 노랑
            else:
                color = (255, 0, 0)    # 파랑

            xs = [lm.x * img_w for lm in landmarks]
            ys = [lm.y * img_h for lm in landmarks]
            min_x, max_x = int(min(xs)), int(max(xs))
            min_y, max_y = int(min(ys)), int(max(ys))

            # 얼굴 박스
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

            # 표정 텍스트
            cv2.putText(frame, expression, (min_x, min_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            break  # max_num_faces=1 이라 첫 얼굴만

    cv2.imshow("Expression Detection (v4 - MediaPipe)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
face_mesh.close()
cv2.destroyAllWindows()
