import math
import cv2
import mediapipe as mp
import pyttsx3

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
PRESENCE_THRESHOLD = 0.5
VISIBILITY_THRESHOLD = 0.5
# For static images
IMAGE_FILES = []
say = pyttsx3.init()
cont = 1


def _normalized_to_pixel_coordinates(
        normalized_x, normalized_y, image_width,
        image_height):
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                          math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def draw1(img, angle, std):
    # print(angle, per)
    per = angle / std
    y = int(100 / per)
    # Check for the dumbbell curls
    color = (255, 0, 255) if (std - 10 <= angle <= std + 15) else (0, 255, 0)

    if y > 650:
        y = 650
    elif y < 100:
        y = 100
    cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
    cv2.rectangle(img, (1100, y), (1175, 650), color, cv2.FILLED)

    cv2.putText(img, f'{("perfect" if (std - 10 <= angle <= std + 15) else "nope")}', (1000, 75),
                cv2.FONT_HERSHEY_PLAIN, 2,
                color, 2)


def draw2(img, angle, std, x, name):
    # print(angle, per)
    per = angle / std
    y = int(100 / per)

    if y > 650:
        y = 650
    elif y < 100:
        y = 100
    # Check for the dumbbell curls
    color = (255, 0, 255) if (std - 10 <= angle <= std + 15) else (0, 255, 0)

    cv2.rectangle(img, (x, 100), (x + 75, 650), color, 3)
    cv2.rectangle(img, (x, y), (x + 75, 650), color, cv2.FILLED)

    cv2.putText(img, f'{(f"{name} perfect" if (std - 10 <= angle <= std + 15) else f"{name} nope")} ', (x - 100, 75),
                cv2.FONT_HERSHEY_PLAIN, 2,
                color, 2)


# 计算角度
def cal(p1, p2, p3):
    a = ((p2[0] - p3[0]) ** 2 + (p2[1] - p3[1]) ** 2) ** 0.5
    b = ((p1[0] - p3[0]) ** 2 + (p1[1] - p3[1]) ** 2) ** 0.5
    c = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    # print(a,b,c)
    angle_b = math.degrees(math.acos((b * b - a * a - c * c) / (-2 * a * c)))
    return angle_b


cap = cv2.VideoCapture(0)
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    i = 0
    while cap.isOpened():
        success, image = cap.read()
        image = cv2.resize(image, (1280, 720))

        if not success:
            print("Ignoring empty camera frame.")
            break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_rows, image_cols, _ = image.shape
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        idx_to_coordinates = {}

        if results.pose_landmarks is None:
            continue
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            if ((landmark.HasField('visibility') and
                 landmark.visibility < VISIBILITY_THRESHOLD) or
                    (landmark.HasField('presence') and
                     landmark.presence < PRESENCE_THRESHOLD)):
                continue
            landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                           image_cols, image_rows)
            if landmark_px:
                idx_to_coordinates[idx] = landmark_px
        dic = {}
        angele = {}
        for k in idx_to_coordinates.keys():
            if k in list(range(11, 17)) or k in list(range(23, 29)):
                dic[k] = idx_to_coordinates[k]
        try:  # 预摆阶段
            angele['left_arm'] = cal(dic[13], dic[11], dic[23])
            angele['left_elbow'] = cal(dic[15], dic[13], dic[11])
            left_elbow = cal(dic[15], dic[13], dic[11])
            left_arm = angele['left_arm']
            left_arm = cal(dic[13], dic[11], dic[23])
            # 手臂与手腕髋
            distance = (
                               ((dic[23][0] - dic[15][0]) ** 2 + (dic[23][1] - dic[15][1]) ** 2) ** 0.5) / (
                               ((dic[15][0] - dic[11][0]) ** 2 + (dic[15][1] - dic[11][1]) ** 2) ** 0.5)
            # 髋、踝与手臂
            distance2 = abs(dic[23][0] - dic[27][0]) / (
                    ((dic[15][0] - dic[11][0]) ** 2 + (dic[15][1] - dic[11][1]) ** 2) ** 0.5)
            is_line = cal(dic[23], dic[25], dic[27])
            print(distance, is_line)
            if is_line > 165 and distance2 > 0.3:
                if cont % 2 == 0:
                    words = '离 地'
                    say.say(words)
                    say.runAndWait()
                angele['left_knee'] = cal(dic[23], dic[25], dic[27])
                left_knee = cal(dic[23], dic[25], dic[27])
                cv2.putText(image, f'already jump', (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (19, 0, 200), 3)
                draw1(image, left_knee, 165)
                cv2.imwrite(f'{i}.jpg', image)
            elif is_line > 165 and distance > 0.3 and dic[23][0] > dic[15][0]:
                words = '前摆'
                if cont % 23 == 0:
                    say.say(words)
                    say.runAndWait()
                print(words)
                cv2.putText(image, f'front sweep', (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (19, 0, 200), 3)
                draw1(image, left_arm, 160)
                print(words)
                draw1(image, left_arm, 160)
                cv2.imwrite(f'{i}.jpg', image)
            elif is_line > 165 and distance > 0.3 and dic[23][0] + 60 < dic[15][0]:
                words = '后摆'
                if cont % 23 == 0:
                    say.say(words)
                    say.runAndWait()
                print(words)
                cv2.putText(image, f'back sweep', (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (19, 0, 200), 3)
                cv2.imwrite(f'{i}.jpg', image)
                draw1(image, left_arm, 70)
            elif cal(dic[11], dic[23], dic[25]) < 150:
                words = '预跳'
                if cont % 24 == 0:
                    say.say(words)
                    say.runAndWait()
                    print(words)
                cv2.putText(image, f'Prepare jump', (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (19, 0, 200), 3)
                cv2.imwrite(f'{i}.jpg', image)
                left_hip = cal(dic[11], dic[23], dic[25])
                left_knee = cal(dic[23], dic[25], dic[27])
                left_arm = cal(dic[13], dic[11], dic[23])
                draw2(image, left_arm, 90, 800, 'arm')
                draw2(image, left_hip, 52, 1000, 'hip')
                draw2(image, left_knee, 55, 1200, 'knee')
                cv2.imwrite(f'{i}.jpg', image)
            cont += 1
            i += 1
        except:
            pass
        cv2.imshow('MediaPipe Pose', image)
        i += 1
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
