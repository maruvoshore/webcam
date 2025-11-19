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
    - mouth_ratio: 입이 얼마나 벌어졌는지
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

    # 랜드마크 좌표 픽셀로 변환
    lx, ly = to_pixel(LEFT_MOUTH)
    rx, ry = to_pixel(RIGHT_MOUTH)
    ux, uy = to_pixel(UPPER_LIP)
    bx, by = to_pixel(LOWER_LIP)

    # 입 가로/세로 길이
    mouth_width = get_distance((lx, ly), (rx, ry))
    mouth_height = get_distance((ux, uy), (bx, by))

    if mouth_width == 0:
        return "NEUTRAL"

    # 입이 얼마나 벌어졌는지 (세로/가로 비율)
    mouth_ratio = mouth_height / mouth_width

    # 입술 중앙 y좌표 (윗입술과 아랫입술 사이)
    center_y = (uy + by) / 2
    # 입꼬리 평균 y좌표
    corner_y = (ly + ry) / 2

    # 입꼬리가 중앙보다 얼마나 위에 있는지
    # (위로 갈수록 y값이 작으니 center_y - corner_y 사용)
    corner_lift = (center_y - corner_y) / mouth_width  # 값 클수록 입꼬리가 위로

    # -----------------------------
    # 튜닝 가능한 기준값들
    # -----------------------------
    # 입꼬리 기준
    LIFT_SMILE = 0.012       # 이 이상이면 "입꼬리가 꽤 올라간 미소"
    LIFT_NEUTRAL_MAX = 0.004 # 이 이하면 "거의 무표정"

    # 입 벌림 기준
    BIG_MOUTH = 0.22         # 이 이상이면 입을 크게 벌린 상태 (치아 보이는 수준)
    SMALL_MOUTH = 0.04       # 이 이하면 입을 거의 안 벌린 상태

    # 1) 입꼬리가 충분히 위로 올라가 있으면 → SMILE (입 다문 미소 포함)
    if corner_lift > LIFT_SMILE:
        return "SMILE"

    # 2) 입도 거의 안 벌어지고, 입꼬리도 거의 안 올라갔으면 → NEUTRAL
    if mouth_ratio < SMALL_MOUTH and corner_lift < LIFT_NEUTRAL_MAX:
        return "NEUTRAL"

    # 3) 입을 많이 벌렸지만, 입꼬리가 위로 약간이라도 올라간 경우 → SMILE (큰 웃음)
    if mouth_ratio > BIG_MOUTH and corner_lift > LIFT_NEUTRAL_MAX:
        return "SMILE"

    # 4) 나머지 애매한 구간은 일단 NEUTRAL 처리
    return "NEUTRAL"


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 거울 모드
    frame = cv2.flip(frame, 1)
    img_h, img_w, _ = frame.shape

    # Mediapipe는 RGB 입력 사용
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

            # 얼굴 전체를 감싸는 박스 (대충 bounding box)
            xs = [lm.x * img_w for lm in landmarks]
            ys = [lm.y * img_h for lm in landmarks]
            min_x, max_x = int(min(xs)), int(max(xs))
            min_y, max_y = int(min(ys)), int(max(ys))

            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

            cv2.putText(frame, expression, (min_x, min_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            break  # max_num_faces=1 이라 첫 번째 얼굴만 처리

    cv2.imshow("Expression Detection (v4.2 - MediaPipe)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
face_mesh.close()
cv2.destroyAllWindows()