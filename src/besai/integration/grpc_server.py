import grpc
from concurrent import futures
from besai.integration import besai_service_pb2, besai_service_pb2_grpc
import os
from besai_system import BeSAISystem
import json

import logging
from besai.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class BeSAIServicer(besai_service_pb2_grpc.BeSAIServiceServicer):
    def __init__(self):
        self.besai = BeSAISystem("localhost:50051", "localhost:9092")

    def ProcessInput(self, request, context):
        logger.info(f"ProcessInput called with input: {request.input}")
        output, analysis, hypothesis = self.besai.process_input(request.input)
        logger.info(f"ProcessInput output: {output}")
        return besai_service_pb2.ProcessInputResponse(
            output=output,
            analysis=json.dumps(analysis),
            hypothesis=json.dumps(hypothesis)
        )

    def ExploreGRPC(self, request, context):
        logger.info(f"ExploreGRPC called with topic: {request.topic}")
        result, analysis, insights = self.besai.explore_topic(request.topic)
        logger.info(f"ExploreGRPC result: {result}")
        return besai_service_pb2.ExploreResponse(
            result=result,
            analysis=json.dumps(analysis),
            insights=insights
        )

    def GenerateInsight(self, request, context):
        logger.info(f"GenerateInsight called with topic: {request.topic}")
        insight = self.besai.reasoning_engine.generate_insight(request.topic)
        logger.info(f"GenerateInsight result: {insight}")
        return besai_service_pb2.InsightResponse(insight=insight)

    def PerformReasoning(self, request, context):
        logger.info(f"PerformReasoning called with query: {request.query}")
        result = self.besai.reasoning_engine.reason(request.query)
        logger.info(f"PerformReasoning result: {result}")
        return besai_service_pb2.ReasoningResponse(result=result)

    def QueryKnowledgeBase(self, request, context):
        logger.info(f"QueryKnowledgeBase called with query: {request.query}")
        result = self.besai.query_knowledge_base(request.query)
        logger.info(f"QueryKnowledgeBase result: {result}")
        return besai_service_pb2.QueryResponse(result=json.dumps(result))

    def SetAlteredState(self, request, context):
        logger.info(f"SetAlteredState called with state: {request.state}")
        result = self.besai.set_altered_state(request.state)
        logger.info(f"SetAlteredState result: {result}")
        return besai_service_pb2.AlteredStateResponse(result=result)

    def SetReasoningParameters(self, request, context):
        logger.info(f"SetReasoningParameters called with focus_level: {request.focus_level}, associative_thinking: {request.associative_thinking}")
        self.besai.reasoning_engine.set_focus_level(request.focus_level)
        self.besai.reasoning_engine.set_associative_thinking(request.associative_thinking)
        result = f"Reasoning parameters updated. Focus level: {request.focus_level}, Associative thinking: {request.associative_thinking}"
        logger.info(f"SetReasoningParameters result: {result}")
        return besai_service_pb2.ReasoningParametersResponse(result=result)

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
