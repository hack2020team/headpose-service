import cv2
import numpy as np

from .mark_detector import MarkDetector
from .pose_estimator import PoseEstimator
from .stabilizer import Stabilizer

CNN_INPUT_SIZE = 128


class FrameProcessor(object):
    def __init__(self, frame_width, frame_height):
        self.mark_detector = MarkDetector()
        self.pose_estimator = PoseEstimator(
            img_size=(frame_height, frame_width),
        )
        self.pose_stabilizers = [
            Stabilizer(
                state_num=2,
                measure_num=1,
                cov_process=0.1,
                cov_measure=0.1,
            )
            for _ in range(6)
        ]

    def process_frame(self, frame):
        frame_data = frame.frame_data

        nparr = np.fromstring(frame_data, np.uint8)
        frame_data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        facebox = self.mark_detector.extract_cnn_facebox(frame_data)

        if facebox is None:
            # TODO(Sam): return explicit error
            return None

        face_img = frame_data[
            facebox[1]:facebox[3],
            facebox[0]:facebox[2],
        ]

        face_image = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
        face_image = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        marks = self.mark_detector.detect_marks([face_image])

        marks *= (facebox[2] - facebox[0])
        marks[:, 0] += facebox[0]
        marks[:, 1] += facebox[1]
        pose = self.pose_estimator.solve_pose_by_68_points(marks)
        steady_pose = []
        pose_np = np.array(pose).flatten()
        for value, ps_stb in zip(pose_np, self.pose_stabilizers):
            # NOTE: temporal requirement here?
            ps_stb.update([value])
            steady_pose.append(ps_stb.state[0])
        steady_pose = np.reshape(steady_pose, (-1, 3))

        return steady_pose
