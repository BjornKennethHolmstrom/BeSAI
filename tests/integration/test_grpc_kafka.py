import unittest
import grpc
from besai.integration import besai_service_pb2, besai_service_pb2_grpc
from besai.integration.kafka_utils import get_producer, get_consumer, send_message, consume_messages
from besai.integration.grpc_server import serve
import os
import time
import threading

class TestGRPCKafka(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Start the gRPC server in a separate thread
        cls.server = serve()
        cls.server_thread = threading.Thread(target=cls.server.wait_for_termination)
        cls.server_thread.start()
        time.sleep(2)  # Give the server some time to start

    @classmethod
    def tearDownClass(cls):
        # Stop the gRPC server
        cls.server.stop(0)
        cls.server_thread.join()

    def test_grpc_service(self):
        with grpc.insecure_channel(os.getenv('GRPC_SERVER_ADDRESS', 'localhost:50051')) as channel:
            stub = besai_service_pb2_grpc.BeSAIServiceStub(channel)
            response = stub.ProcessInput(besai_service_pb2.InputRequest(input="test"))
            self.assertIsNotNone(response.output)
            self.assertNotEqual(response.output, "")

    def test_kafka(self):
        producer = get_producer()
        consumer = get_consumer('test_topic')
        
        test_message = {"key": "value"}
        send_message(producer, 'test_topic', test_message)
        
        start_time = time.time()
        timeout = 10  # 10 seconds timeout
        received_message = None
        
        while time.time() - start_time < timeout:
            messages = list(consumer.poll(timeout_ms=1000).values())
            if messages:
                received_message = messages[0][0].value
                break
        
        self.assertEqual(received_message, test_message)

if __name__ == '__main__':
    unittest.main()
