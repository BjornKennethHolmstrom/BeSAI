# BeSAI/src/self_improvement/code_analyzer.py

import re
import ast
import astunparse
from typing import List, Dict, Any
from collections import defaultdict

# for CodeTester
# import ast
import copy
from io import StringIO
import sys
import contextlib

class CodeImprover(ast.NodeTransformer):
    def __init__(self):
        self.changes_made = False

    def visit_For(self, node):
        # List comprehension transformation
        if (len(node.body) == 1 and isinstance(node.body[0], ast.Expr) 
            and isinstance(node.body[0].value, ast.Call)
            and isinstance(node.body[0].value.func, ast.Attribute) 
            and node.body[0].value.func.attr == 'append'):
            
            target = node.target
            iter = node.iter
            elt = node.body[0].value.args[0]
            new_node = ast.Assign(
                targets=[node.body[0].value.func.value],
                value=ast.ListComp(elt=elt, generators=[ast.comprehension(target=target, iter=iter, ifs=[], is_async=0)])
            )
            self.changes_made = True
            return new_node

        # Enumerate transformation
        if (isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) 
            and node.iter.func.id == 'range' and len(node.iter.args) == 1):
            
            iterable = node.iter.args[0]
            new_node = ast.For(
                target=ast.Tuple(elts=[node.target, ast.Name(id='item', ctx=ast.Store())], ctx=ast.Store()),
                iter=ast.Call(func=ast.Name(id='enumerate', ctx=ast.Load()), args=[iterable], keywords=[]),
                body=node.body,
                orelse=node.orelse
            )
            self.changes_made = True
            return new_node

        return self.generic_visit(node)

    def visit_BinOp(self, node):
        # f-string transformation
        if isinstance(node.op, ast.Mod) and isinstance(node.left, ast.Str):
            template = node.left.s
            if isinstance(node.right, ast.Tuple):
                values = []
                parts = template.split('%')
                for i, part in enumerate(parts):
                    values.append(ast.Constant(value=part.replace('%d', '').replace('%s', '')))
                    if i < len(node.right.elts):
                        values.append(ast.FormattedValue(value=node.right.elts[i], conversion=-1, format_spec=None))
            else:
                values = [
                    ast.Constant(value=template.split('%')[0].replace('%d', '').replace('%s', '')),
                    ast.FormattedValue(value=node.right, conversion=-1, format_spec=None)
                ]
            new_node = ast.JoinedStr(values=values)
            self.changes_made = True
            return new_node
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        # Type hint transformation
        if not node.returns:
            # Attempt to infer the return type
            return_type = self.infer_return_type(node)
            node.returns = ast.Name(id=return_type, ctx=ast.Load())
            self.changes_made = True
        return self.generic_visit(node)

    def infer_return_type(self, node):
        # This is a simple inference. You can make it more sophisticated.
        if any(isinstance(n, ast.Return) for n in ast.walk(node)):
            return_node = next(n for n in ast.walk(node) if isinstance(n, ast.Return))
            if isinstance(return_node.value, ast.Num):
                return 'int' if isinstance(return_node.value.n, int) else 'float'
            elif isinstance(return_node.value, ast.Str):
                return 'str'
        return 'Any'  # Default to Any if we can't infer

    def visit_If(self, node):
        # Ternary operator transformation
        if (len(node.body) == 1 and len(node.orelse) == 1
            and isinstance(node.body[0], ast.Assign) and isinstance(node.orelse[0], ast.Assign)
            and node.body[0].targets[0].id == node.orelse[0].targets[0].id):
            
            new_node = ast.Assign(
                targets=node.body[0].targets,
                value=ast.IfExp(
                    test=node.test,
                    body=node.body[0].value,
                    orelse=node.orelse[0].value
                )
            )
            self.changes_made = True
            return new_node
        return self.generic_visit(node)

class PatternLearner:
    def __init__(self):
        self.learned_patterns: List[Dict] = []
        self.pattern_counts = defaultdict(int)

    def learn_from_example(self, before: str, after: str, description: str):
        generalized_before = self.generalize_pattern(before)
        generalized_after = self.generalize_pattern(after)

        pattern = {
            'before': generalized_before,
            'after': generalized_after,
            'description': description
        }

        if pattern not in self.learned_patterns:
            self.learned_patterns.append(pattern)
            print(f"Learned new pattern: {description}")
            print(f"Generalized Before: {generalized_before}")
            print(f"Generalized After: {generalized_after}")
        else:
            print(f"Pattern already known: {description}")

        self.pattern_counts[f"{generalized_before}->{generalized_after}"] += 1

    def generalize_pattern(self, pattern: str) -> str:
        # Replace specific variable names with placeholders
        return re.sub(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', r'{\1}', pattern)

    def apply_learned_patterns(self, code: str) -> str:
        for pattern in sorted(self.learned_patterns, key=lambda p: self.pattern_counts[f"{p['before']}->{p['after']}"], reverse=True):
            before = pattern['before']
            after = pattern['after']
            
            # Replace placeholders with regex capture groups
            regex_pattern = re.sub(r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}', r'([a-zA-Z_][a-zA-Z0-9_]*)', before)
            
            def replacement_func(match):
                result = after
                for i, group in enumerate(match.groups(), 1):
                    result = result.replace(f'{{{match.group(i)}}}', group)
                return result

            new_code = re.sub(regex_pattern, replacement_func, code)
            if new_code != code:
                print(f"Applied pattern: {pattern['description']}")
                code = new_code

        return code

    def learn_from_user_feedback(self, original_code: str, improved_code: str, user_feedback: str):
        if "better" in user_feedback.lower():
            self.learn_from_example(original_code, improved_code, "Learned from positive user feedback")
        elif "worse" in user_feedback.lower():
            print("Noted negative feedback. Will avoid this type of change in the future.")

class CodeAnalyzer:
    def __init__(self):
        self.pattern_learner = PatternLearner()
        self.code_tester = CodeTester()

        self.built_in_improvements = [
            (r'for (\w+) in range\(len\((\w+)\)\):\s*(\w+)\.append\(([^)]+)\)', 
             r'\3 = [\4 for \1 in range(len(\2))]'),
            (r'return "([^"]+)"\s*%\s*(.+)', r"return f'\1'.format(\2)"),
            (r'def (\w+)\(([^)]*)\):', r'def \1(\2) -> Any:')
        ]

    def improve_code(self, code: str) -> str:
        original_code = code
        print("Applying built-in improvements:")
        for pattern, replacement in self.built_in_improvements:
            new_code = re.sub(pattern, replacement, code)
            if new_code != code:
                print(f"Applied built-in improvement")
                code = new_code
        
        print("Applying learned patterns:")
        code = self.pattern_learner.apply_learned_patterns(code)
        
        improved_code = code
        
        # Test the improvements
        test_results = self.code_tester.test_function(original_code, improved_code)
        self.code_tester.report_results()
        
        # Learn from test results
        self.learn_from_test_results(test_results)
        
        return improved_code

    def learn_from_test_results(self, test_results):
        for result in test_results:
            if not result['is_equal']:
                print(f"Warning: Improvement changed behavior for function {result['function']}")
                print(f"Test case: {result['test_case']}")
                print(f"Original result: {result['original_result']}")
                print(f"Improved result: {result['improved_result']}")

    def learn_new_pattern(self, before: str, after: str, description: str):
        self.pattern_learner.learn_from_example(before, after, description)

    def provide_feedback(self, original_code: str, improved_code: str, feedback: str):
        self.pattern_learner.learn_from_user_feedback(original_code, improved_code, feedback)

class CodeTester:
    def __init__(self):
        self.test_results = []

    @contextlib.contextmanager
    def capture_stdout(self):
        new_out = StringIO()
        old_out = sys.stdout
        try:
            sys.stdout = new_out
            yield new_out
        finally:
            sys.stdout = old_out

    def generate_test_cases(self, func_def):
        """Generate simple test cases based on function signature."""
        func_name = func_def.name
        args = [arg.arg for arg in func_def.args.args]
        
        test_cases = []
        for i in range(3):  # Generate 3 test cases
            args_values = [f"test_value_{j}_{i}" for j in range(len(args))]
            test_case = f"{func_name}({', '.join(args_values)})"
            test_cases.append(test_case)
        
        return test_cases

    def execute_safely(self, code, globals_dict):
        try:
            exec(code, globals_dict)
        except Exception as e:
            return f"Error: {str(e)}"

    def test_function(self, original_code, improved_code):
        original_globals = {}
        improved_globals = {}
        
        # Execute the original and improved code
        self.execute_safely(original_code, original_globals)
        self.execute_safely(improved_code, improved_globals)
        
        # Parse the improved code to get function definitions
        improved_ast = ast.parse(improved_code)
        func_defs = [node for node in improved_ast.body if isinstance(node, ast.FunctionDef)]
        
        for func_def in func_defs:
            func_name = func_def.name
            if func_name in original_globals and func_name in improved_globals:
                test_cases = self.generate_test_cases(func_def)
                
                for test_case in test_cases:
                    with self.capture_stdout() as original_out:
                        original_result = eval(test_case, original_globals)
                    
                    with self.capture_stdout() as improved_out:
                        improved_result = eval(test_case, improved_globals)
                    
                    is_equal = (original_result == improved_result and 
                                original_out.getvalue() == improved_out.getvalue())
                    
                    self.test_results.append({
                        'function': func_name,
                        'test_case': test_case,
                        'original_result': original_result,
                        'improved_result': improved_result,
                        'is_equal': is_equal
                    })
        
        return self.test_results

    def report_results(self):
        print("\nTest Results:")
        for result in self.test_results:
            print(f"Function: {result['function']}")
            print(f"Test Case: {result['test_case']}")
            print(f"Original Result: {result['original_result']}")
            print(f"Improved Result: {result['improved_result']}")
            print(f"Results Match: {'Yes' if result['is_equal'] else 'No'}")
            print("---")
