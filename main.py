import math
import cv2
import mediapipe as mp
import pyttsx3
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
PRESENCE_THRESHOLD = 0.5
VISIBILITY_THRESHOLD = 0.5

# 标准动作角度
QIANBAI_KNEE = 165  # 前摆，髋、膝、踝的角度
QITIAO_KNEE = 0.3 #起跳下蹲的程度
QITIAO_ARM = 0.3 #起跳手臂后摆的程度

# For static images
IMAGE_FILES = []
# say = pyttsx3.init()
count = 1   #帧数
"""
todo 补充标记对应关键点表格

"""


cap = cv2.VideoCapture(0)   #0表示第一个摄像头，1表示第二个


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


# draw1表示只包含单个变量的动作计算分数并打印，todo 计算逻辑需要梳理
#img：柱状图形
#angle:角度
# std：标准角度
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


#  画出分数柱状图
def draw_score(img, angle, std_min, std_max):
    # todo
    return img

# 画评分柱状
# 需要写注释
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


# 计算三点夹角，顶点为p2
def cal_angle(p1, p2, p3):  # p1为A，p2为B，p3为C
    # BC距离
    a = ((p2[0] - p3[0]) ** 2 + (p2[1] - p3[1]) ** 2) ** 0.5
    # AC距离
    b = ((p1[0] - p3[0]) ** 2 + (p1[1] - p3[1]) ** 2) ** 0.5
    # AB距离
    c = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    # 算出AB，BC夹角，顶点B的角度
    angle_b = math.degrees(math.acos((b * b - a * a - c * c) / (-2 * a * c)))
    return angle_b


# todo 计算跳远距离
def cal_distance():
    jumping_distance = 0
    return jumping_distance


# todo 输出动作分析报告
def jumping_pose_report():
    sns.set_theme(style="ticks", color_codes=True)

    # 加载数据
    df = sns.load_dataset('iris', data_home='seaborn-data', cache=True)

    # 绘图显示
    sns.catplot(x=df["species"], y=df["sepal_length"], data=df)
    plt.show()


# 在图像上显示中文字符
def cv2_add_chinese_text(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型       
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontstyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontstyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=2,     #2代表更heavy模型，性能效果有待观察
        enable_segmentation=True) as pose:
    jumping_no = 1
    while cap.isOpened():
        success, image = cap.read()
        image = cv2.resize(image, (1280, 720))  #720P
        image_no = 1
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
            #print(results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.NOSE])
            if landmark_px:
                idx_to_coordinates[idx] = landmark_px
        dic = {}
        angele = {}
        words = ''  #初始化
        for k in idx_to_coordinates.keys():  # 取下一个图片
            if k in list(range(11, 17)) or k in list(range(23, 29)):
                dic[k] = idx_to_coordinates[k]
        try:
            # 预摆阶段
            angele['left_arm'] = cal_angle(dic[13], dic[11], dic[23])
            angele['left_elbow'] = cal_angle(dic[15], dic[13], dic[11])
            left_elbow = cal_angle(dic[15], dic[13], dic[11])
            left_arm = angele['left_arm']
            # left_arm = cal_angle(dic[13], dic[11], dic[23]) # 没有用，删除
            # 手臂与手腕髋的比值 todo 不知道什么意思
            #腕髋距离/手臂长度，比值越大表示向后伸展距离越大
            distance = (
                               ((dic[23][0] - dic[15][0]) ** 2 + (dic[23][1] - dic[15][1]) ** 2) ** 0.5) / (
                               ((dic[15][0] - dic[11][0]) ** 2 + (dic[15][1] - dic[11][1]) ** 2) ** 0.5)
            # 手臂长度
            left_arm_length = ((dic[15][0] - dic[11][0]) ** 2 + (dic[15][1] - dic[11][1]) ** 2) ** 0.5
            # 髋、踝的横坐标之差/手臂长度，比值越大说明下蹲越明显
            distance2 = abs(dic[23][0] - dic[27][0]) / left_arm_length

            pose_leg = cal_angle(dic[23], dic[25], dic[27])  # 髋、膝、踝的角度，腿的姿态
            print(distance, distance2, pose_leg)
            words = ''  #初始化
            if pose_leg > QIANBAI_KNEE and distance2 > 0.3:  # todo 为什么0.3
                if count % 2 == 0:    # todo 为什么双数帧图片？
                    words = '离地'
                    # say.say(words)
                    # say.runAndWait()
                    print(words)
                angele['left_knee'] = cal_angle(dic[23], dic[25], dic[27])
                left_knee = cal_angle(dic[23], dic[25], dic[27])
                cv2.putText(image, f'already jump', (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (19, 0, 200), 3)
                # cv2_add_chinese_text(image, "起跳", (100, 60), (19, 0, 200), 1.2)
                draw1(image, left_knee, QIANBAI_KNEE)
                cv2.imwrite(f'./images/{jumping_no}_{image_no}.jpg', image)
            elif pose_leg > QIANBAI_KNEE and distance > 0.3 and dic[23][0] > dic[15][0]:
                if count % 23 == 0:
                    words = '前摆'
                    # say.say(words)
                    # say.runAndWait()
                    print(words)
                cv2.putText(image, f'front sweep', (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (19, 0, 200), 3)
                # cv2_add_chinese_text(image, "前摆姿势", (100, 60), (19, 0, 200), 1.2)
                draw1(image, left_arm, 160)
                print(words)
                draw1(image, left_arm, 160)
                cv2.imwrite(f'./images/{jumping_no}_{image_no}.jpg', image)
            elif pose_leg > 165 and distance > 0.3 and dic[23][0] + 60 < dic[15][0]:
                if count % 23 == 0:
                    words = '后摆'
                    # say.say(words)
                    # say.runAndWait()
                    print(words)
                cv2.putText(image, f'back sweep', (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (19, 0, 200), 3)
                draw1(image, left_arm, 70)
                # cv2_add_chinese_text(image, "后摆姿势", (100, 60), (19, 0, 200), 1.2)
                cv2.imwrite(f'./images/{jumping_no}_{image_no}.jpg', image)

            elif cal_angle(dic[11], dic[23], dic[25]) < 150:
                if count % 24 == 0:
                    words = '预跳'
                    # say.say(words)
                    # say.runAndWait()
                    print(words)
                cv2.putText(image, f'prepare jumping', (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (19, 0, 200), 3)
                # cv2_add_chinese_text(image, "预跳姿势", (100, 60), (19, 0, 200), 1.2)
                #cv2.imwrite(f'./images/{image_no}.jpg', image)    #todo。。不知道为什么要多出来这个？
                left_hip = cal_angle(dic[11], dic[23], dic[25])
                left_knee = cal_angle(dic[23], dic[25], dic[27])
                left_arm = cal_angle(dic[13], dic[11], dic[23])
                draw2(image, left_arm, 90, 800, 'arm1')
                draw2(image, left_hip, 52, 1000, 'hip')
                draw2(image, left_knee, 55, 1200, 'knee')
                cv2.imwrite(f'./images/{jumping_no}_{image_no}.jpg', image)
            #else:
            #    image_no -= 1   #解决输出图片序号会跳问题
            count += 1
            image_no += 1    # 另外一张图片
            print(count)
            print(f'./images/{jumping_no}_{image_no}.jpg')
        except (KeyError, ValueError):
            pass
        cv2.imshow('MediaPipe Pose', image)
        if words == '预跳':    #如果没有跳完，次数不加
            jumping_no += 1
        if cv2.waitKey(5) & 0xFF == 27:  # 显示画面，如果按ESC退出画面
            break
cap.release()
