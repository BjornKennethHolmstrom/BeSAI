# BeSAI/src/ethics_safety/explainer.py

from typing import Any, Dict, List

class DecisionNode:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.children: List[DecisionNode] = []

    def add_child(self, child: 'DecisionNode'):
        self.children.append(child)

class DecisionTree:
    def __init__(self):
        self.root: DecisionNode = None

    def set_root(self, root: DecisionNode):
        self.root = root

    def explain(self) -> str:
        if not self.root:
            return "No decision process to explain."
        return self._explain_node(self.root, 0)

    def _explain_node(self, node: DecisionNode, depth: int) -> str:
        indent = "  " * depth
        explanation = f"{indent}{node.name}: {node.description}\n"
        for child in node.children:
            explanation += self._explain_node(child, depth + 1)
        return explanation

class Explainer:
    def __init__(self):
        self.decision_tree = DecisionTree()

    def record_decision(self, name: str, description: str, parent: DecisionNode = None) -> DecisionNode:
        node = DecisionNode(name, description)
        if not self.decision_tree.root:
            self.decision_tree.set_root(node)
        elif parent:
            parent.add_child(node)
        return node

    def explain_decision(self) -> str:
        return self.decision_tree.explain()

# Example usage
explainer = Explainer()

def complex_decision(data: Dict[str, Any]) -> bool:
    root = explainer.record_decision("Initial Evaluation", "Evaluating input data")

    if data["value"] > 100:
        node = explainer.record_decision("High Value Check", "Value exceeds threshold", root)
        if data["risk"] < 0.5:
            explainer.record_decision("Acceptable Risk", "Risk is within acceptable range", node)
            return True
        else:
            explainer.record_decision("High Risk", "Risk exceeds acceptable threshold", node)
            return False
    else:
        node = explainer.record_decision("Low Value Check", "Value is below threshold", root)
        if data["priority"] == "high":
            explainer.record_decision("High Priority", "Item has high priority despite low value", node)
            return True
        else:
            explainer.record_decision("Low Priority", "Item has low priority and low value", node)
            return False

# Test the explainer
test_data = {"value": 150, "risk": 0.4, "priority": "low"}
result = complex_decision(test_data)
print(f"Decision result: {result}")
print("\nDecision Explanation:")
print(explainer.explain_decision())
