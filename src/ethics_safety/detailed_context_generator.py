# BeSAI/src/ethics_safety/detailed_context_generator.py

import random
from typing import Dict, Any, List, Tuple

class DetailedContextGenerator:
    def __init__(self):
        self.action_types = [
            ("Share user data with third party", [
                "user has explicitly given permission",
                "for improving service quality",
                "for marketing purposes",
                "required by law enforcement"
            ]),
            ("Implement new feature", [
                "that collects more user data",
                "that improves user privacy",
                "that may exclude some users",
                "that is highly requested by users"
            ]),
            ("Change pricing", [
                "to increase profitability",
                "to make service more accessible",
                "based on user behavior data",
                "to match competitors"
            ]),
            ("Respond to customer complaint", [
                "by offering compensation",
                "by improving the product",
                "by explaining company policy",
                "by ignoring it due to high volume"
            ]),
            ("Launch marketing campaign", [
                "targeting vulnerable populations",
                "using user data for personalization",
                "promoting sustainable practices",
                "comparing product to competitors"
            ]),
            ("Write blog post", [
                "critizising without reason",
                "critizising with reasons",
                "about own ideas",
                "without sufficient knowledge about subject"
            ])
        ]
        
        self.stakeholders = [
            "Users", "Employees", "Shareholders", "Community",
            "Environment", "Competitors", "Government", "Suppliers", "Family", "Friends"
        ]
        
        self.impact_areas = [
            "Privacy", "Security", "Financial", "Emotional",
            "Physical", "Social", "Environmental", "Legal", "Spiritual"
        ]

    def generate_context(self) -> Dict[str, Any]:
        action, condition = self._generate_conditional_action()
        context = {
            'action': f"{action} {condition}",
            'stakeholders': self._generate_stakeholder_impact(),
            'short_term_impact': self._generate_impact(),
            'long_term_impact': self._generate_impact(),
            'legal_compliance': random.choice([True, False]),
            'transparency': random.randint(1, 10),
            'resource_usage': {
                'financial': random.randint(1, 10),
                'human': random.randint(1, 10),
                'environmental': random.randint(1, 10)
            },
            'alignment_with_values': random.randint(1, 10),
            'potential_risks': self._generate_potential_risks(),
            'potential_benefits': self._generate_potential_benefits(),
            'ethical_dilemmas': self._generate_ethical_dilemmas()
        }
        return context

    def _generate_conditional_action(self) -> Tuple[str, str]:
        action, conditions = random.choice(self.action_types)
        condition = random.choice(conditions)
        return action, condition

    def _generate_stakeholder_impact(self) -> Dict[str, int]:
        return {stakeholder: random.randint(-5, 5) for stakeholder in random.sample(self.stakeholders, 4)}

    def _generate_impact(self) -> Dict[str, int]:
        return {area: random.randint(-5, 5) for area in random.sample(self.impact_areas, 4)}

    def _generate_potential_risks(self) -> List[str]:
        risks = [
            "Data breach", "Reputation damage", "Financial loss",
            "Legal action", "Employee dissatisfaction", "Customer churn",
            "Environmental damage", "Health and safety issues"
        ]
        return random.sample(risks, random.randint(1, 3))

    def _generate_potential_benefits(self) -> List[str]:
        benefits = [
            "Increased revenue", "Improved user satisfaction", "Cost reduction",
            "Competitive advantage", "Improved brand image", "Innovation",
            "Improved employee satisfaction", "Positive environmental impact"
        ]
        return random.sample(benefits, random.randint(1, 3))

    def _generate_ethical_dilemmas(self) -> List[str]:
        dilemmas = [
            "Privacy vs Convenience", "Profit vs Social Responsibility",
            "Innovation vs Stability", "Transparency vs Confidentiality",
            "Short-term gains vs Long-term sustainability",
            "Individual benefit vs Collective good",
            "Efficiency vs Employment", "Progress vs Tradition"
        ]
        return random.sample(dilemmas, random.randint(1, 2))

context_generator = DetailedContextGenerator()

if __name__ == "__main__":
    # Test the context generator
    context = context_generator.generate_context()
    print("Generated Context:")
    for key, value in context.items():
        print(f"{key}: {value}")
