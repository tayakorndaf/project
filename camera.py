from itertools import count
from typing import Counter
from urllib import request
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
        np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle


counter_Squat = 0
stage_Squat = None


class Squatt(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        global counter_Squat
        global stage_Squat
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while self.video.isOpened():
                ret, frame = self.video.read()

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame.flags.writeable = False

                results = pose.process(frame)

                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                try:
                    landmarks = results.pose_landmarks.landmark

                    # Get coordinates
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                    # Calculate angle
                    angle = calculate_angle(right_hip, right_knee, right_ankle)

                    # Curl counter logic
                    if angle > 160:
                        stage_Squat = "up"
                    if angle < 95 and stage_Squat == 'up':
                        stage_Squat = "down"
                        counter_Squat += 1

                except:
                    pass

                cv2.rectangle(frame, (1, 1), (73, 73), (255, 255, 255), -1)
                cv2.putText(frame, str(counter_Squat),
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)

                # Render detections
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(
                                              color=(245, 117, 66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(
                                              color=(245, 66, 230), thickness=2, circle_radius=2)
                                          )

                ret, jpg = cv2.imencode('.jpg', frame)

                return jpg.tobytes(), counter_Squat


counter_leglungess = 0
counter_leglungess1 = 0
stage_leglungess = None
stage_leglungess2 = None


class leglungess(object):

    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        global counter_leglungess
        global stage_leglungess

        global counter_leglungess1
        global stage_leglungess1
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while self.video.isOpened():
                ret, frame = self.video.read()

                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    # Get coordinates
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    # Calculate angle
                    angle = calculate_angle(left_hip, left_knee, left_ankle)

                    if angle > 150:
                        stage_leglungess = "up"
                    if angle < 100 and stage_leglungess == 'up':
                        stage_leglungess = "down"
                        counter_leglungess += 1
                        print(f"{counter_leglungess} left")

                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                    angle = calculate_angle(right_hip, right_knee, right_ankle)

                    if angle > 150:
                        stage_leglungess1 = "up"
                    if angle < 100 and stage_leglungess1 == 'up':
                        stage_leglungess1 = "down"
                        counter_leglungess1 += 1
                        print(f"{counter_leglungess1} right")
                except:
                    pass

                cv2.rectangle(frame, (1, 1), (73, 73), (255, 255, 255), -1)
                cv2.putText(frame, str(counter_leglungess),
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, 'left',
                            (70, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 0, cv2.LINE_AA)

                cv2.putText(frame, 'right',
                            (70, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 0, cv2.LINE_AA)
                cv2.rectangle(frame, (1, 150), (73, 73), (255, 255, 255), -1)
                cv2.putText(frame, str(counter_leglungess1),
                            (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)

                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(
                                              color=(245, 117, 66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(
                                              color=(245, 66, 230), thickness=2, circle_radius=2)
                                          )

                ret, jpg = cv2.imencode('.jpg', frame)
                return jpg.tobytes(), counter_leglungess1


counter_toysoldier = 0
counter_toysoldier1 = 0
stage_toysoldier = None
stage_toysoldier1 = None


class toysoldierr(object):

    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        global counter_toysoldier
        global counter_toysoldier1
        global stage_toysoldier
        global stage_toysoldier1

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while self.video.isOpened():
                ret, frame = self.video.read()

                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    # Get coordinates
                    left_shouder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                    # Calculate angle
                    angle = calculate_angle(left_shouder, left_hip, left_knee)

                    if angle > 150:
                        stage_toysoldier = "down"
                    if angle < 110 and stage_toysoldier == 'down':
                        stage_toysoldier = "up"
                        counter_toysoldier += 1
                        print(f"{counter_toysoldier} left")

                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                    angle = calculate_angle(
                        right_shoulder, right_hip, right_ankle)

                    if angle > 150:
                        stage_toysoldier1 = "down"
                    if angle < 110 and stage_toysoldier1 == 'down':
                        stage_toysoldier1 = "up"
                        counter_toysoldier1 += 1
                        print(f"{counter_toysoldier1} right")

                except:
                    pass

                cv2.rectangle(frame, (1, 1), (73, 73), (255, 255, 255), -1)
                cv2.putText(frame, str(counter_toysoldier),
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, 'left',
                            (70, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 0, cv2.LINE_AA)

                cv2.putText(frame, 'right',
                            (70, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 0, cv2.LINE_AA)
                cv2.rectangle(frame, (1, 150), (73, 73), (255, 255, 255), -1)
                cv2.putText(frame, str(counter_toysoldier1),
                            (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)

                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(
                                              color=(245, 117, 66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(
                                              color=(245, 66, 230), thickness=2, circle_radius=2)
                                          )
                ret, jpg = cv2.imencode('.jpg', frame)
                return jpg.tobytes(), counter_toysoldier1


counter_Highkneeruns = 0
counter_Highkneeruns1 = 0
stage_Highkneeruns = None
stage_Highkneeruns1 = None


class Highkneeruns(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        global counter_Highkneeruns
        global counter_Highkneeruns1
        global stage_Highkneeruns
        global stage_Highkneeruns1
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while self.video.isOpened():
                ret, frame = self.video.read()

                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    # Get coordinates
                    left_shouder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                    # Calculate angle
                    angle = calculate_angle(left_shouder, left_hip, left_knee)

                    # Curl counter logic
                    if angle > 150:
                        stage_Highkneeruns = "down"
                    if angle < 50 and stage_Highkneeruns == 'down':
                        stage_Highkneeruns = "up"
                        counter_Highkneeruns += 1
                        print(f"{counter_Highkneeruns} left")

                    right_shouder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                     landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

                    angle = calculate_angle(
                        right_shouder, right_hip, right_knee)

                    # Curl counter logic
                    if angle > 150:
                        stage_Highkneeruns1 = "down"
                    if angle < 50 and stage_Highkneeruns1 == 'down':
                        stage_Highkneeruns1 = "up"
                        counter_Highkneeruns1 += 1
                        print(f"{counter_Highkneeruns1} right")
                except:
                    pass
                cv2.rectangle(frame, (1, 1), (73, 73), (255, 255, 255), -1)
                cv2.putText(frame, str(counter_Highkneeruns),
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, 'left',
                            (70, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 0, cv2.LINE_AA)

                cv2.putText(frame, 'right',
                            (70, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 0, cv2.LINE_AA)
                cv2.rectangle(frame, (1, 150), (73, 73), (255, 255, 255), -1)
                cv2.putText(frame, str(counter_Highkneeruns1),
                            (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)

                # Stage data
                cv2.putText(frame, 'STAGE', (550, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, stage_toysoldier,
                            (550, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(
                                              color=(245, 117, 66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(
                                              color=(245, 66, 230), thickness=2, circle_radius=2)
                                          )
                ret, jpg = cv2.imencode('.jpg', frame)
                return jpg.tobytes(), counter_Highkneeruns1


counter_standingsidecrunch = 0
counter_standingsidecrunch1 = 0
stage_standingsidecrunch = 0
stage_standingsidecrunch1 = 0


class standingsidecrunch(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        global counter_standingsidecrunch
        global counter_standingsidecrunch1
        global stage_standingsidecrunch
        global stage_standingsidecrunch1
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while self.video.isOpened():
                ret, frame = self.video.read()

                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    # Get coordinates
                    left_shouder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                    # Calculate angle
                    angle = calculate_angle(left_shouder, left_hip, left_knee)

                    # Curl counter logic
                    if angle > 150:
                        stage_standingsidecrunch = "down"
                    if angle < 60 and stage_standingsidecrunch == 'down':
                        stage_standingsidecrunch = "up"
                        counter_standingsidecrunch += 1
                        print(counter_standingsidecrunch)

                    right_shouder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                     landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

                    # Calculate angle
                    angle = calculate_angle(
                        right_shouder, right_hip, right_knee)

                    # Curl counter logic
                    if angle > 150:
                        stage_standingsidecrunch1 = "down"
                    if angle < 60 and stage_standingsidecrunch1 == 'down':
                        stage_standingsidecrunch1 = "up"
                        counter_standingsidecrunch1 += 1
                        print(f"{counter_standingsidecrunch1} right")

                except:
                    pass

                cv2.rectangle(frame, (1, 1), (73, 73), (255, 255, 255), -1)
                cv2.putText(frame, str(counter_standingsidecrunch),
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, 'left',
                            (70, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 0, cv2.LINE_AA)

                cv2.putText(frame, 'right',
                            (70, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 0, cv2.LINE_AA)
                cv2.rectangle(frame, (1, 150), (73, 73), (255, 255, 255), -1)
                cv2.putText(frame, str(counter_standingsidecrunch1),
                            (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)

                # Stage data
                cv2.putText(frame, 'STAGE', (550, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, stage_toysoldier,
                            (550, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
                # Render detections
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(
                                              color=(245, 117, 66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(
                                              color=(245, 66, 230), thickness=2, circle_radius=2)
                                          )

                ret, jpg = cv2.imencode('.jpg', frame)
                return jpg.tobytes(), counter_standingsidecrunch1


counter_jumpslap = 0
stage_jumpslap = None


class jumpslap(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        global counter_jumpslap
        global stage_jumpslap
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while self.video.isOpened():
                ret, frame = self.video.read()

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                results = pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                try:
                    landmarks = results.pose_landmarks.landmark

                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

                    angle = calculate_angle(
                        left_hip, left_shoulder, left_elbow)
                   
                    if angle > 160:
                        stage_jumpslap = "up"
                    if angle < 30 and stage_jumpslap == 'up':
                        stage_jumpslap = "down"
                        counter_jumpslap += 1
                        print(counter_jumpslap)

                except:
                    pass

                cv2.rectangle(frame, (1, 1), (73, 73), (255, 255, 255), -1)
                cv2.putText(frame, str(counter_jumpslap),
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)

                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(
                                              color=(245, 117, 66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(
                                              color=(245, 66, 230), thickness=2, circle_radius=2)
                                          )
                ret, jpg = cv2.imencode('.jpg', frame)
                return jpg.tobytes(), counter_jumpslap


class Video(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()
        ret, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()
