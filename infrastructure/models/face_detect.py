import logging

import cv2
import mediapipe as mp
import onnxruntime as ort
from insightface.app import FaceAnalysis
from scipy.spatial import distance as dist


class FaceDetect():
    def __init__(self):
        available_providers = ort.get_available_providers()
        logging.info(f"Available providers: {available_providers}")

        self.model = self.load_face_analysis_model()
        # self.face_mesh = self.load_mediapipe_face_mesh()

    def load_face_analysis_model(self):
        model = FaceAnalysis(providers=['CPUExecutionProvider'])
        model.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id=-1 表示使用CPU
        # model = FaceAnalysis(providers=['CoreMLExecutionProvider'])
        # model.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=-1 表示使用CPU
        logging.info("FaceAnalysis model initialized")
        return model

    def load_mediapipe_face_mesh(self):
        # 初始化mediapipe的面部检测器
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        logging.info("Mediapipe face mesh initialized")
        return face_mesh

    def eye_aspect_ratio(self, eye_landmarks):
        # 计算眼睛的长宽比
        A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def count_face_ratio_and_eye_state(self, img):
        # 进行人脸检测
        faces = self.model.get(img)
        ratio = 0
        ear = 0

        if len(faces) == 0:
            # 没有检测到人脸
            logging.info("没有检测到人脸")
            return ratio, ear

        # 检测到至少一张人脸，找出占比最大的人脸
        max_face = max(faces, key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))
        bbox = max_face.bbox.astype(int)  # 获取占比最大的人脸的边界框

        # 确保边界框在图像范围内
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)  # img.shape[0] 是图像的高度，img.shape[1] 是图像的宽度

        # 裁剪人脸区域前检查边界框是否有效
        if x2 > x1 and y2 > y1:
            cropped_face = img[y1:y2, x1:x2]  # 裁剪人脸区域
            # 再次检查裁剪后的图像是否为空
            if cropped_face.size <= 0:
                logging.warning("无效的边界框")
                return ratio, ear

            ratio = (x2 - x1) * (y2 - y1) / (img.shape[0] * img.shape[1])
            logging.info(f"最大人脸比例为: {ratio}")

            # 使用mediapipe检测关键点
            results = self.face_mesh.process(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # 提取左眼和右眼的关键点
                    left_eye_indices = [33, 160, 158, 133, 153, 144]
                    right_eye_indices = [362, 385, 387, 263, 373, 380]

                    left_eye = [(int(face_landmarks.landmark[i].x * cropped_face.shape[1]),
                                 int(face_landmarks.landmark[i].y * cropped_face.shape[0])) for i in
                                left_eye_indices]
                    right_eye = [(int(face_landmarks.landmark[i].x * cropped_face.shape[1]),
                                  int(face_landmarks.landmark[i].y * cropped_face.shape[0])) for i in
                                 right_eye_indices]

                    # 计算眼睛的长宽比
                    left_ear = self.eye_aspect_ratio(left_eye)
                    right_ear = self.eye_aspect_ratio(right_eye)

                    # 眼睛长宽比的平均值
                    ear = (left_ear + right_ear) / 2.0

                    # 判断眼睛是否睁开
                    eye_open = ear > 0.2  # 这个阈值可以根据需要调整
                    logging.info(f"眼睛是否睁开: {eye_open}")

            else:
                logging.warning("裁剪的人脸图像为空")

        return ratio, ear

    def count_max_face_emb(self, img):
        # 进行人脸检测
        faces = self.model.get(img)
        # print(faces)

        if len(faces) == 0:
            # 没有检测到人脸
            logging.info("没有检测到人脸")
            return None

        # print("face already detect")
        # # 检测到至少一张人脸，找出占比最大的人脸
        # max_face = max(faces, key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))
        # face_vector = max_face.embedding
        # 检测到至少一张人脸，找出占比最大的人脸
        if len(faces) == 1:
            # 将对齐后的图像转换为特征向量
            face_vector = faces[0].embedding
        elif len(faces) > 1:
            # 找到面积最大的人脸
            largest_face = max(faces,
                               key=lambda face: abs((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])))
            # 将对齐后的图像转换为特征向量
            face_vector = largest_face.embedding
        # print('face_vector', face_vector)

        return face_vector

    def detect_and_crop_face(self, img):
        logging.info("Detecting and cropping face")
        faces = self.model.get(img)
        if len(faces) == 0:
            logging.info("No faces detected")
            return None

        max_area = 0
        max_face = None
        for face in faces:
            bbox = face.bbox.astype(int)
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area > max_area:
                max_area = area
                max_face = face

        if max_face is not None:
            bbox = max_face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)

            if x2 > x1 and y2 > y1:
                cropped_face = img[y1:y2, x1:x2]
                if cropped_face.size > 0:
                    logging.info(f"Face detected with bounding box: {bbox}")
                    return cropped_face
        logging.info("No valid face detected")
        return None
