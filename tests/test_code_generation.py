import unittest
from src.self_improvement.code_generation import CodeGeneration

class TestCodeGeneration(unittest.TestCase):

    def setUp(self):
        self.code_gen = CodeGeneration()

    def test_generate_code(self):
        feature_request = "Create a function named 'example_feature' that takes two parameters"
        generated_code = self.code_gen.generate_code(feature_request)
        self.assertIn("def example_feature(param1, param2):", generated_code)
        self.assertIn("Implement the functionality here", generated_code)
        self.assertIn("return result", generated_code)

    def test_parse_feature_request(self):
        feature_request = "Create a function named 'test_function' that takes three parameters"
        components = self.code_gen._parse_feature_request(feature_request)
        self.assertEqual(components["type"], "function")
        self.assertEqual(components["name"], "example_feature")  # This will fail, showing the need for proper NLP
        self.assertEqual(len(components["parameters"]), 2)  # This will fail, showing the need for proper NLP

    def test_create_code_structure(self):
        components = {
            "type": "function",
            "name": "test_function",
            "parameters": ["a", "b", "c"],
            "functionality": "Test functionality"
        }
        structure = self.code_gen._create_code_structure(components)
        self.assertEqual(structure.name, "test_function")
        self.assertEqual(len(structure.args.args), 3)

    def test_implement_code(self):
        from ast import FunctionDef, arguments, arg, Pass
        structure = FunctionDef(
            name="test_function",
            args=arguments(args=[arg(arg="a"), arg(arg="b")], posonlyargs=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=[Pass()],
            decorator_list=[]
        )
        implemented_code = self.code_gen._implement_code(structure)
        self.assertIn("def test_function(a, b):", implemented_code)
        self.assertIn("Implement the functionality here", implemented_code)
        self.assertIn("return result", implemented_code)

    def test_learn_and_apply_pattern(self):
        pattern = "def {function_name}({params}):\n    return {param1} + {param2}"
        self.code_gen.learn_pattern("simple_addition", pattern)
        applied_code = self.code_gen.apply_pattern("simple_addition", function_name="add_numbers", params=["x", "y"])
        self.assertEqual(applied_code, "def add_numbers(x, y):\n    return x + y")

    def test_get_source_code(self):
        def sample_function(a, b):
            return a + b
        
        source_code = CodeGeneration.get_source_code(sample_function)
        self.assertIn("def sample_function(a, b):", source_code)
        self.assertIn("return a + b", source_code)

    def test_unknown_pattern(self):
        with self.assertRaises(ValueError):
            self.code_gen.apply_pattern("unknown_pattern")

    def test_unsupported_feature_type(self):
        components = {
            "type": "unsupported_type",
            "name": "test",
            "parameters": [],
            "functionality": "Test"
        }
        with self.assertRaises(ValueError):
            self.code_gen._create_code_structure(components)

if __name__ == '__main__':
    unittest.main()
