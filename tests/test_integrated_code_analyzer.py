import unittest
import re
from src.self_improvement.code_analyzer import CodeAnalyzer

class TestIntegratedCodeAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = CodeAnalyzer()

    def test_basic_improvement(self):
        original_code = """
def greet(name):
    return "Hello, %s!" % name
"""
        improved_code = self.analyzer.improve_code(original_code)
        self.assertTrue(
            re.search(r"return f['\"]Hello,\s*{name}!['\"]", improved_code) or
            re.search(r"return ['\"](Hello,\s*){name}!['\"]\.format\(name=name\)", improved_code) or
            re.search(r"return f['\"]Hello, %s!['\"]\.format\(name\)", improved_code)
        )

    def test_code_generation_improvement(self):
        original_code = """
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
"""
        improved_code = self.analyzer.improve_code(original_code)
        self.assertIn("return sum(numbers)", improved_code)

    def test_ast_based_improvement(self):
        original_code = """
def create_list(n):
    result = []
    for i in range(n):
        result.append(i * 2)
    return result
"""
        improved_code = self.analyzer.improve_code(original_code)
        self.assertIn("return [(i * 2) for i in range(n)]", improved_code)

    def test_combined_improvements(self):
        original_code = """
def process_data(data):
    result = []
    for i in range(len(data)):
        if data[i] % 2 == 0:
            result.append("Even: %d" % data[i])
        else:
            result.append("Odd: %d" % data[i])
    return result
"""
        improved_code = self.analyzer.improve_code(original_code)
        self.assertTrue(re.search(r"for\s*((\()?i,\s*item(\))?\s+in enumerate\(data\)|\w+\s+in enumerate\(data\)):", improved_code))
        self.assertTrue(re.search(r"(f['\"]Even:\s*{[^}]+}['\"]|['\"]Even: %d['\"] % item)", improved_code))
        self.assertTrue(re.search(r"(f['\"]Odd:\s*{[^}]+}['\"]|['\"]Odd: %d['\"] % item)", improved_code))


    def test_code_behavior_preservation(self):
        original_code = """
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)
"""
        improved_code = self.analyzer.improve_code(original_code)
        
        # Execute both versions and compare results
        exec(original_code, globals())
        original_result = factorial(5)
        
        exec(improved_code, globals())
        improved_result = factorial(5)
        
        self.assertEqual(original_result, improved_result)

if __name__ == '__main__':
    unittest.main()
