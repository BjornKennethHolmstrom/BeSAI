import ast
import astunparse
import inspect

class CodeGeneration:
    def __init__(self):
        self.known_patterns = {}

    def generate_code(self, feature_request, original_code=None):
        if original_code is None:
            return self._generate_generic_code(feature_request)
        
        tree = ast.parse(original_code)
        
        if "use f-string" in feature_request.lower():
            tree = self._improve_string_formatting(tree)
        if "use list comprehension" in feature_request.lower():
            tree = self._improve_list_creation(tree)
        if "optimize sum" in feature_request.lower():
            tree = self._optimize_sum(tree)
        if "use enumerate" in feature_request.lower():
            tree = self._use_enumerate(tree)
        
        return astunparse.unparse(tree)

    def _generate_generic_code(self, feature_request):
        return f"\ndef improved_function(param1, param2):\n    # TODO: Implement {feature_request}\n    pass\n"

    def _parse_feature_request(self, feature_request):
        """
        Parse the feature request into components.
        
        :param feature_request: A string describing the desired feature
        :return: A dictionary of feature components
        """
        # This is a placeholder implementation. In a real system, this would use
        # NLP techniques to break down the request into actionable components.
        components = {
            "type": "function",
            "name": "example_feature",
            "parameters": ["param1", "param2"],
            "functionality": "Perform some operation on parameters"
        }
        return components

    def _create_code_structure(self, feature_components):
        """
        Create an AST structure based on feature components.
        
        :param feature_components: A dictionary of feature components
        :return: An AST node representing the code structure
        """
        if feature_components["type"] == "function":
            func_def = ast.FunctionDef(
                name=feature_components["name"],
                args=ast.arguments(
                    args=[ast.arg(arg=param) for param in feature_components["parameters"]],
                    posonlyargs=[],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[]
                ),
                body=[ast.Pass()],  # Placeholder, will be replaced in _implement_code
                decorator_list=[]
            )
            return func_def
        else:
            raise ValueError(f"Unsupported feature type: {feature_components['type']}")

    def _implement_code(self, code_structure):
        """
        Implement the actual code based on the code structure.
        
        :param code_structure: An AST node representing the code structure
        :return: Implemented code as a string
        """
        if isinstance(code_structure, ast.FunctionDef):
            # Replace the placeholder body with actual implementation
            code_structure.body = [
                ast.Expr(ast.Str(s="Implement the functionality here")),
                ast.Return(ast.Name(id="result", ctx=ast.Load()))
            ]
        
        # Convert the AST back to source code
        return astunparse.unparse(code_structure)

    def learn_pattern(self, pattern_name, code_snippet):
        """
        Learn a new code pattern from a given snippet.
        
        :param pattern_name: A string naming the pattern
        :param code_snippet: A string containing the code to learn from
        """
        self.known_patterns[pattern_name] = code_snippet

    def apply_pattern(self, pattern_name, **kwargs):
        """
        Apply a known pattern with given parameters.
        
        :param pattern_name: The name of the pattern to apply
        :param kwargs: Parameters to fill in the pattern
        :return: The applied pattern as a string
        """
        if pattern_name not in self.known_patterns:
            raise ValueError(f"Unknown pattern: {pattern_name}")
        
        pattern = self.known_patterns[pattern_name]
        
        # Handle the 'params' argument specially
        if 'params' in kwargs:
            params = kwargs['params']
            kwargs['params'] = ', '.join(params)
            for i, param in enumerate(params, start=1):
                kwargs[f'param{i}'] = param
        
        return pattern.format(**kwargs)

    def _improve_string_formatting(self, tree):
        class StringFormattingTransformer(ast.NodeTransformer):
            def visit_BinOp(self, node):
                if isinstance(node.op, ast.Mod) and isinstance(node.left, ast.Str):
                    format_spec = ast.Str(s='')
                    if isinstance(node.right, ast.Name):
                        value = node.right
                    else:
                        value = node.right.elts[0] if isinstance(node.right, ast.Tuple) else node.right
                    return ast.JoinedStr([
                        ast.Constant(value=node.left.s.replace('%s', '{').replace('%d', '{}')),
                        ast.FormattedValue(
                            value,
                            conversion=-1,
                            format_spec=format_spec
                        )
                    ])
                return node

            def visit_Call(self, node):
                if isinstance(node.func, ast.Attribute) and node.func.attr == 'format':
                    if isinstance(node.func.value, ast.Str):
                        return ast.JoinedStr([
                            ast.Constant(value=node.func.value.s.replace('{}', '{')),
                            ast.FormattedValue(
                                node.args[0],
                                conversion=-1,
                                format_spec=None
                            )
                        ])
                return node

            def visit_Return(self, node):
                if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Mod):
                    return ast.Return(value=self.visit_BinOp(node.value))
                elif isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'format':
                    return ast.Return(value=self.visit_Call(node.value))
                return node

        return StringFormattingTransformer().visit(tree)

    def _improve_list_creation(self, tree):
        class ListComprehensionTransformer(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                new_body = []
                for stmt in node.body:
                    if isinstance(stmt, ast.For) and len(stmt.body) == 1 and isinstance(stmt.body[0], ast.Expr) and isinstance(stmt.body[0].value, ast.Call) and isinstance(stmt.body[0].value.func, ast.Attribute) and stmt.body[0].value.func.attr == 'append':
                        new_body.append(ast.Return(
                            value=ast.ListComp(
                                elt=stmt.body[0].value.args[0],
                                generators=[
                                    ast.comprehension(
                                        target=stmt.target,
                                        iter=stmt.iter,
                                        ifs=[],
                                        is_async=0
                                    )
                                ]
                            )
                        ))
                    else:
                        new_body.append(stmt)
                node.body = new_body
                return node

        return ListComprehensionTransformer().visit(tree)

    def _optimize_sum(self, tree):
        class SumOptimizer(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                if len(node.body) >= 2 and isinstance(node.body[0], ast.Assign) and isinstance(node.body[1], ast.For):
                    for_loop = node.body[1]
                    if len(for_loop.body) == 1 and isinstance(for_loop.body[0], ast.AugAssign) and isinstance(for_loop.body[0].op, ast.Add):
                        return ast.FunctionDef(
                            name=node.name,
                            args=node.args,
                            body=[
                                ast.Return(
                                    value=ast.Call(
                                        func=ast.Name(id='sum', ctx=ast.Load()),
                                        args=[for_loop.iter],
                                        keywords=[]
                                    )
                                )
                            ],
                            decorator_list=node.decorator_list,
                            returns=None
                        )
                return node

        return SumOptimizer().visit(tree)

    def _use_enumerate(self, tree):
        class EnumerateTransformer(ast.NodeTransformer):
            def visit_For(self, node):
                if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id in ['range', 'len']:
                    iter_arg = node.iter.args[0]
                    if isinstance(iter_arg, ast.Call) and isinstance(iter_arg.func, ast.Name) and iter_arg.func.id == 'len':
                        iter_arg = iter_arg.args[0]
                    return ast.For(
                        target=ast.Tuple([ast.Name(id='i', ctx=ast.Store()), ast.Name(id='item', ctx=ast.Store())], ctx=ast.Store()),
                        iter=ast.Call(
                            func=ast.Name(id='enumerate', ctx=ast.Load()),
                            args=[iter_arg],
                            keywords=[]
                        ),
                        body=self.update_body(node.body),
                        orelse=node.orelse
                    )
                return node

            def update_body(self, body):
                class BodyTransformer(ast.NodeTransformer):
                    def visit_Subscript(self, node):
                        if isinstance(node.value, ast.Name) and node.value.id == 'data' and isinstance(node.slice, ast.Name) and node.slice.id == 'i':
                            return ast.Name(id='item', ctx=node.ctx)
                        return node

                return [BodyTransformer().visit(stmt) for stmt in body]

        return EnumerateTransformer().visit(tree)

    @staticmethod
    def get_source_code(obj):
        """
        Get the source code of a given object.
        
        :param obj: The object to get the source code from
        :return: The source code as a string
        """
        return inspect.getsource(obj)
