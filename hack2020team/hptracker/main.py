from concurrent import futures
import grpc

from ..headposeservice import pose_service_pb2_grpc
from ..headposeservice.pose_service_pb2 import PoseResponse
from .process_frame import FrameProcessor

FRAME_WIDTH = 640
FRAME_HEIGHT = 480


class HeadPoseServicer(pose_service_pb2_grpc.HeadPoseApiServicer):
    def __init__(self):
        self.frame_processor = FrameProcessor(
            frame_width=FRAME_WIDTH,
            frame_height=FRAME_HEIGHT,
        )
        super().__init__()

    def GetPose(self, frame, context):
        result = self.frame_processor.process_frame(frame)
        print(result)
        return PoseResponse(frame_identifier=frame.frame_identifier)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pose_service_pb2_grpc.add_HeadPoseApiServicer_to_server(
      HeadPoseServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()
