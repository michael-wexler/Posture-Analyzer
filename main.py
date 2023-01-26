import cv2
import time
import math
import mediapipe as mp

#function to calculate angle between two vectors <x2 - x1, y2 - y1> and <x3 - x1, -y1>
def calculate_angle(x1, y1, x2, y2):
    theta = math.acos((y2 - y1) * (-y1) / (math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    angle = math.degrees(theta)
    return angle

def alert():
    pass

font = cv2.FONT_HERSHEY_DUPLEX
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
yellow = (0, 255, 255)
white = (255, 255, 255)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
good_frames = 0
bad_frames = 0

class Camera(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.good_frames = 0
        self.bad_frames = 0

    def __del__(self):
        self.cap.release()

    #function that gets analytics from frames and display data in real-time
    def get_frame(self):
        success, image = self.cap.read()
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) * (4 / 5))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * (4 / 5))
        image = cv2.resize(image, (width, height), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        h, w = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        keypoints = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        lm = keypoints.pose_landmarks
        lmPose = mp_pose.PoseLandmark

        #mediapipe pose estimate for left side of body
        left_shoulder_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
        left_shoulder_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
        left_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
        left_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
        left_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
        left_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

        #calculating angles based on indices of body parts
        neck_angle = calculate_angle(left_shoulder_x, left_shoulder_y, left_ear_x, left_ear_y)
        back_angle = calculate_angle(left_hip_x, left_hip_y, left_shoulder_x, left_shoulder_y)

        #labeling each body part with a dot
        cv2.circle(image, (left_shoulder_x, left_shoulder_y), 7, yellow, -1)
        cv2.circle(image, (left_ear_x, left_ear_y), 7, yellow, -1)
        cv2.circle(image, (left_shoulder_x, left_shoulder_y - 100), 7, yellow, -1)
        cv2.circle(image, (left_hip_x, left_hip_y), 7, yellow, -1)
        cv2.circle(image, (left_hip_x, left_hip_y - 100), 7, yellow, -1)

        #displaying angles for neck and back
        angle_text_string = 'Neck: ' + str(int(neck_angle)) + ' Back: ' + str(int(back_angle))

        #threshold for having "good posture" (based on trial and error)
        if neck_angle < 35 and back_angle < 10:
            self.good_frames += 1
            cv2.putText(image, angle_text_string, (10, 30), font, 0.9, green, 2)
            cv2.putText(image, str(int(neck_angle)), (left_shoulder_x + 10, left_shoulder_y), font, 0.9, green, 2)
            cv2.putText(image, str(int(back_angle)), (left_hip_x + 10, left_hip_y), font, 0.9, green, 2)
            cv2.line(image, (left_shoulder_x, left_shoulder_y), (left_ear_x, left_ear_y), green, 4)
            cv2.line(image, (left_shoulder_x, left_shoulder_y), (left_shoulder_x, left_shoulder_y - 100), green, 4)
            cv2.line(image, (left_hip_x, left_hip_y), (left_shoulder_x, left_shoulder_y), green, 4)
            cv2.line(image, (left_hip_x, left_hip_y), (left_hip_x, left_hip_y - 100), green, 4)
        else:
            self.bad_frames += 1
            cv2.putText(image, angle_text_string, (10, 30), font, 0.9, red, 2)
            cv2.putText(image, str(int(neck_angle)), (left_shoulder_x + 10, left_shoulder_y), font, 0.9, red, 2)
            cv2.putText(image, str(int(back_angle)), (left_hip_x + 10, left_hip_y), font, 0.9, red, 2)
            cv2.line(image, (left_shoulder_x, left_shoulder_y), (left_ear_x, left_ear_y), red, 4)
            cv2.line(image, (left_shoulder_x, left_shoulder_y), (left_shoulder_x, left_shoulder_y - 100), red, 4)
            cv2.line(image, (left_hip_x, left_hip_y), (left_shoulder_x, left_shoulder_y), red, 4)
            cv2.line(image, (left_hip_x, left_hip_y), (left_hip_x, left_hip_y - 100), red, 4)

        #calculating number of frames labeled good/bad and percentage of good frames
        good_time = (1 / fps) * self.good_frames
        bad_time = (1 / fps) * self.bad_frames
        total_time = good_time + bad_time
        correct_percent = (good_time / total_time) * 100

        #displaying total duration for good/bad posture
        time_string_good = 'Good Posture Duration: ' + str(round(good_time, 1)) + ' seconds'
        cv2.putText(image, time_string_good, (10, h - 45), font, 0.9, white, 2)
        time_string_bad = 'Bad Posture Duration: ' + str(round(bad_time, 1)) + ' seconds'
        cv2.putText(image, time_string_bad, (10, h - 15), font, 0.9, white, 2)
        time_string_total = 'Total Duration: ' + str(round(total_time, 1)) + ' seconds'
        cv2.putText(image, time_string_total, (10, h - 75), font, 0.9, white, 2)
        percent_string = str(round(correct_percent, 2)) + ' %'
        cv2.putText(image, percent_string, (w - 160, h - 75), font, 1.2, dark_blue, 2)

        #notify user for extended time with bad posture
        #if bad_time > 10:
            #alert()
        ret, jpeg = cv2.imencode('.jpg', image)
        return [jpeg.tobytes(), good_time, bad_time]