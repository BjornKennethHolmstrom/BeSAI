import logging
import grpc
from besai.integration import besai_service_pb2, besai_service_pb2_grpc
from besai.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = besai_service_pb2_grpc.BeSAIServiceStub(channel)
        
        logger.info("Calling ProcessInput")
        response = stub.ProcessInput(besai_service_pb2.InputRequest(input="What is the nature of consciousness?"))
        print("ProcessInput Response:")
        print(response.output)
        
        logger.info("Calling ExploreGRPC")
        response = stub.ExploreGRPC(besai_service_pb2.ExploreRequest(topic="artificial intelligence"))
        print("\nExploreGRPC Response:")
        print(response.result)
        
        logger.info("Calling GenerateInsight")
        response = stub.GenerateInsight(besai_service_pb2.InsightRequest(topic="quantum physics"))
        print("\nGenerateInsight Response:")
        print(response.insight)
        
        logger.info("Calling PerformReasoning")
        response = stub.PerformReasoning(besai_service_pb2.ReasoningRequest(query="How does climate change affect biodiversity?"))
        print("\nPerformReasoning Response:")
        print(response.result)

if __name__ == '__main__':
    run()
