import grpc
from concurrent import futures
from besai.integration import besai_service_pb2, besai_service_pb2_grpc
import os
from besai.besai_console import BeSAIConsole
import threading

import logging
from besai.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class BeSAIServicer(besai_service_pb2_grpc.BeSAIServiceServicer):
    def __init__(self):
        self.besai = BeSAIConsole()

    def ProcessInput(self, request, context):
        logger.info(f"ProcessInput called with input: {request.input}")
        output = self.besai.default(request.input)
        logger.info(f"ProcessInput output: {output}")
        return besai_service_pb2.OutputResponse(output=output)

    def ExploreGRPC(self, request, context):
        logger.info(f"ExploreGRPC called with topic: {request.topic}")
        result = self.besai.do_explore(request.topic)
        logger.info(f"ExploreGRPC result: {result}")
        return besai_service_pb2.ExploreResponse(result=result)

    def GenerateInsight(self, request, context):
        logger.info(f"GenerateInsight called with topic: {request.topic}")
        insight = self.besai.do_insight(request.topic)
        logger.info(f"GenerateInsight result: {insight}")
        return besai_service_pb2.InsightResponse(insight=insight if insight else "No insight available after exploration.")

    def PerformReasoning(self, request, context):
        logger.info(f"PerformReasoning called with query: {request.query}")
        result = self.besai.do_reason(request.query)
        logger.info(f"PerformReasoning result: {result}")
        return besai_service_pb2.ReasoningResponse(result=result if result else "No reasoning available after exploration.")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    besai_service_pb2_grpc.add_BeSAIServiceServicer_to_server(BeSAIServicer(), server)
    server.add_insecure_port(os.getenv('GRPC_SERVER_ADDRESS', 'localhost:50051'))
    server.start()
    logger.info("gRPC server started on port 50051")
    return server

if __name__ == '__main__':
    server = serve()
    server.wait_for_termination()
