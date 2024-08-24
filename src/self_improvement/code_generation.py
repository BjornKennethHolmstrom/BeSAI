# BeSAI/src/self_improvement/code_generation.py

import ast
import astunparse

class CodeGeneration:
    def __init__(self):
        self.safety_check = SafetyCheck()

    def generate_function(self, function_name, args, body, return_type=None, docstring=None):
        """Generate a Python function as an AST."""
        function_args = ast.arguments(
            args=[ast.arg(arg=arg, annotation=None) for arg in args],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[]
        )

        function_body = ast.parse(body).body

        if docstring:
            function_body.insert(0, ast.Expr(ast.Str(s=docstring)))

        func_ast = ast.FunctionDef(
            name=function_name,
            args=function_args,
            body=function_body,
            decorator_list=[],
            returns=ast.Name(id=return_type, ctx=ast.Load()) if return_type else None
        )
        return func_ast

    def generate_class(self, class_name, methods, attributes=None, base_classes=None, docstring=None):
        """Generate a Python class as an AST."""
        class_body = []

        if docstring:
            class_body.append(ast.Expr(ast.Str(s=docstring)))

        if attributes:
            for attr, value in attributes.items():
                class_body.append(ast.Assign(
                    targets=[ast.Name(id=attr, ctx=ast.Store())],
                    value=ast.parse(value).body[0].value
                ))

        for method in methods:
            class_body.append(self.generate_function(**method))

        class_ast = ast.ClassDef(
            name=class_name,
            bases=[ast.Name(id=base, ctx=ast.Load()) for base in (base_classes or [])],
            keywords=[],
            body=class_body,
            decorator_list=[]
        )
        return class_ast

    def ast_to_code(self, ast_node):
        """Convert an AST to Python code string."""
        return astunparse.unparse(ast_node)

    def generate_and_check_code(self, code_type, **kwargs):
        """Generate code and perform a safety check."""
        if code_type == 'function':
            ast_node = self.generate_function(**kwargs)
        elif code_type == 'class':
            ast_node = self.generate_class(**kwargs)
        else:
            return "Unsupported code type."

        code = self.ast_to_code(ast_node)
        if self.safety_check.check_code(code):
            return code
        else:
            return "Code generation failed safety check."

class SafetyCheck:
    def __init__(self):
        self.forbidden_modules = ['os', 'sys', 'subprocess']
        self.forbidden_functions = ['eval', 'exec', '__import__']

    def check_code(self, code):
        """Perform a basic safety check on generated code."""
        try:
            parsed_ast = ast.parse(code)
            for node in ast.walk(parsed_ast):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        if alias.name in self.forbidden_modules:
                            return False
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in self.forbidden_functions:
                        return False
            return True
        except SyntaxError:
            return False
