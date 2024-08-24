# BeSAI/src/self_improvement/pattern_learner.py

import ast
import astunparse
from typing import List, Dict, Callable
import re

class PatternLearner:
    def __init__(self):
        self.learned_patterns: List[Dict] = []

    def create_transformation_function(self, diff: Dict) -> Callable:
        """Create a function that can apply the learned transformation."""
        def transform(node: ast.AST) -> ast.AST:
            if isinstance(node, ast.Return) and isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Add):
                return ast.Return(
                    value=ast.Call(
                        func=ast.Name(id='sum', ctx=ast.Load()),
                        args=[ast.Tuple(elts=[node.value.left, node.value.right], ctx=ast.Load())],
                        keywords=[]
                    )
                )
            return node
        return transform

    def learn_from_example(self, before: str, after: str, description: str):
        self.learned_patterns.append({
            'before': before,
            'after': after,
            'description': description
        })
        print(f"Learned new pattern: {description}")
        print(f"Before: {before}")
        print(f"After: {after}")

    def compute_ast_diff(self, before_ast: ast.AST, after_ast: ast.AST) -> Dict:
        """Compute the difference between two ASTs."""
        # This is a simplified diff. A more sophisticated diff algorithm would be needed for real-world use.
        before_dump = ast.dump(before_ast)
        after_dump = ast.dump(after_ast)
        
        # Find the first difference
        i = 0
        while i < len(before_dump) and i < len(after_dump) and before_dump[i] == after_dump[i]:
            i += 1
        
        return {
            'before': before_dump[i:i+100],  # Take a snippet of the difference
            'after': after_dump[i:i+100]
        }

    def generate_pattern_key(self, diff: Dict) -> str:
        """Generate a unique key for the learned pattern."""
        return f"{hash(diff['before'])}-{hash(diff['after'])}"

    def create_transformation_function(self, diff: Dict) -> Callable:
        """Create a function that can apply the learned transformation."""
        def transform(node: ast.AST) -> ast.AST:
            node_dump = ast.dump(node)
            if diff['before'] in node_dump:
                new_node_dump = node_dump.replace(diff['before'], diff['after'])
                return ast.parse(astunparse.unparse(ast.literal_eval(new_node_dump)))
            return node
        return transform

    def apply_learned_patterns(self, code: str) -> str:
        tree = ast.parse(code)
        for pattern in self.learned_patterns:
            tree = self.apply_pattern(tree, pattern)
        return astunparse.unparse(tree)

    def apply_pattern(self, tree: ast.AST, pattern: Dict) -> ast.AST:
        class PatternTransformer(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                if pattern['before'] in astunparse.unparse(node):
                    print(f"Applying pattern: {pattern['description']}")
                    new_code = astunparse.unparse(node).replace(pattern['before'], pattern['after'])
                    return ast.parse(new_code).body[0]
                return node

        return PatternTransformer().visit(tree)

    def learn_from_user_feedback(self, original_code: str, improved_code: str, user_feedback: str):
        if "better" in user_feedback.lower():
            self.learn_from_example(original_code, improved_code, "Learned from positive user feedback")
        elif "worse" in user_feedback.lower():
            print("Noted negative feedback. Will avoid this type of change in the future.")
            # Here we could implement logic to avoid similar changes in the future
