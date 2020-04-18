# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

from hack2020team.headposeservice import pose_service_pb2 as hack2020team_dot_headposeservice_dot_pose__service__pb2


class HeadPoseApiStub(object):
    """Missing associated documentation comment in .proto file"""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetPose = channel.unary_unary(
                '/youlearn.headpose.v1.HeadPoseApi/GetPose',
                request_serializer=hack2020team_dot_headposeservice_dot_pose__service__pb2.Frame.SerializeToString,
                response_deserializer=hack2020team_dot_headposeservice_dot_pose__service__pb2.PoseResponse.FromString,
                )


class HeadPoseApiServicer(object):
    """Missing associated documentation comment in .proto file"""

    def GetPose(self, request, context):
        """Missing associated documentation comment in .proto file"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_HeadPoseApiServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetPose': grpc.unary_unary_rpc_method_handler(
                    servicer.GetPose,
                    request_deserializer=hack2020team_dot_headposeservice_dot_pose__service__pb2.Frame.FromString,
                    response_serializer=hack2020team_dot_headposeservice_dot_pose__service__pb2.PoseResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'youlearn.headpose.v1.HeadPoseApi', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class HeadPoseApi(object):
    """Missing associated documentation comment in .proto file"""

    @staticmethod
    def GetPose(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/youlearn.headpose.v1.HeadPoseApi/GetPose',
            hack2020team_dot_headposeservice_dot_pose__service__pb2.Frame.SerializeToString,
            hack2020team_dot_headposeservice_dot_pose__service__pb2.PoseResponse.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)