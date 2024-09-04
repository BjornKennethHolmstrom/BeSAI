import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.enhanced_knowledge_base import EnhancedKnowledgeBase

from typing import List, Dict, Any, Optional

class ReasoningEngine:
    def __init__(self, knowledge_base: EnhancedKnowledgeBase):
        self.kb = knowledge_base
        self.focus_level = 0.5
        self.associative_thinking = 0.5

    def infer_transitive_relationships(self, start_entity: str, relationship_type: str) -> List[Dict[str, Any]]:
        visited = set()
        to_visit = [start_entity]
        inferred_relationships = []

        while to_visit:
            current_entity = to_visit.pop(0)
            if current_entity in visited:
                continue
            visited.add(current_entity)

            relationships = self.kb.get_relationships(current_entity)
            for related_entity, rel_type, attrs in relationships:
                if rel_type == relationship_type:
                    certainty = attrs.get('certainty', 1.0)
                    inferred_relationships.append({
                        "from": start_entity,
                        "to": related_entity,
                        "relationship": relationship_type,
                        "inferred": current_entity != start_entity,
                        "certainty": certainty if current_entity == start_entity else certainty * 0.9  # Reduce certainty for inferred relationships
                    })
                    to_visit.append(related_entity)

        return inferred_relationships

    def find_common_attributes(self, entity1: str, entity2: str) -> Dict[str, Any]:
        attrs1 = self.kb.get_entity(entity1)
        attrs2 = self.kb.get_entity(entity2)
        
        common_attrs = {}
        if attrs1 and attrs2:
            for key, value in attrs1.items():
                if key in attrs2 and attrs2[key] == value:
                    common_attrs[key] = value
        
        return common_attrs

    def suggest_new_relationships(self, entity: str) -> List[Dict[str, Any]]:
        entity_attrs = self.kb.get_entity(entity)
        all_entities = self.kb.graph.nodes()
        suggestions = []

        for other_entity in all_entities:
            if other_entity != entity:
                common_attrs = self.find_common_attributes(entity, other_entity)
                if common_attrs:
                    suggestions.append({
                        "entity1": entity,
                        "entity2": other_entity,
                        "common_attributes": common_attrs,
                        "suggested_relationship": "related_to",
                        "certainty": 0.5  # Suggested relationships have lower certainty
                    })

        return suggestions

    def infer_attribute_from_relationships(self, entity: str, attribute: str) -> Optional[Dict[str, Any]]:
        relationships = self.kb.get_relationships(entity)
        inferred_value = None
        max_certainty = 0

        for related_entity, _, _ in relationships:
            related_attrs = self.kb.get_entity(related_entity)
            if related_attrs and attribute in related_attrs:
                if isinstance(related_attrs['certainty'], dict):
                    certainty = related_attrs['certainty'].get(attribute, 0.5) * 0.8
                else:
                    certainty = related_attrs['certainty'] * 0.8  # Use the global certainty
                if certainty > max_certainty:
                    inferred_value = related_attrs[attribute]
                    max_certainty = certainty

        if inferred_value is not None:
            return {"value": inferred_value, "certainty": max_certainty}
        return None

    def generate_hypothesis(self, entity: str) -> Optional[Dict[str, Any]]:
        entity_attrs = self.kb.get_entity(entity)
        if entity_attrs is None:
            return None

        relationships = self.kb.get_relationships(entity)
        
        hypothesis = {
            "entity": entity,
            "known_attributes": entity_attrs,
            "inferred_attributes": {},
            "potential_relationships": []
        }

        # Infer attributes from relationships
        for related_entity, rel_type, _ in relationships:
            related_attrs = self.kb.get_entity(related_entity)
            if related_attrs:
                for attr, value in related_attrs.items():
                    if attr not in entity_attrs:
                        inferred = self.infer_attribute_from_relationships(entity, attr)
                        if inferred:
                            hypothesis["inferred_attributes"][attr] = inferred

        # Suggest potential relationships
        hypothesis["potential_relationships"] = self.suggest_new_relationships(entity)

        return hypothesis

    def explain_inference(self, entity: str, attribute: str) -> str:
        inferred = self.infer_attribute_from_relationships(entity, attribute)
        if inferred:
            related_entities = [
                related for related, _, _ in self.kb.get_relationships(entity)
                if attribute in self.kb.get_entity(related)
            ]
            explanation = f"The attribute '{attribute}' for '{entity}' was inferred with a certainty of {inferred['certainty']:.2f} "
            explanation += f"based on its relationships with: {', '.join(related_entities)}."
            return explanation
        return f"Unable to infer the attribute '{attribute}' for '{entity}'."

    def set_focus_level(self, level: float):
        self.focus_level = max(0.0, min(1.0, level))

    def set_associative_thinking(self, level: float):
        self.associative_thinking = max(0.0, min(1.0, level))

    def reset_parameters(self):
        self.focus_level = 0.5
        self.associative_thinking = 0.5

    def generate_insight(self, topic: str) -> str:
        # This is a placeholder implementation
        entities = self.kb.query({"type": topic})
        if entities:
            entity = random.choice(entities)
            return f"Insight on {topic}: {entity['attributes'].get('description', 'No description available.')}"
        return f"No insight available for {topic}"

    def calculate_relevance(self, topic1: str, topic2: str) -> float:
        try:
            # Get the relationships for each topic
            relationships1 = self.kb.get_relationships(topic1)
            relationships2 = self.kb.get_relationships(topic2)

            # Extract only the related entity names
            entities1 = set(rel[0] for rel in relationships1 if isinstance(rel, tuple) and len(rel) > 0)
            entities2 = set(rel[0] for rel in relationships2 if isinstance(rel, tuple) and len(rel) > 0)

            # Calculate the Jaccard similarity between the two sets of entities
            common_entities = entities1.intersection(entities2)
            union_entities = entities1.union(entities2)

            if not union_entities:
                return 0.0  # If there are no entities, return 0 relevance

            jaccard_similarity = len(common_entities) / len(union_entities)

            # Incorporate the individual relevance scores from the knowledge base
            kb_relevance1 = self.kb.calculate_relevance(topic1)
            kb_relevance2 = self.kb.calculate_relevance(topic2)

            # Combine Jaccard similarity with knowledge base relevance scores
            combined_relevance = (jaccard_similarity + kb_relevance1 + kb_relevance2) / 3

            return combined_relevance

        except Exception as e:
            print(f"Error calculating relevance between {topic1} and {topic2}: {str(e)}")
            return 0.0  # Return 0 relevance if there's an error

# Example usage
if __name__ == "__main__":
    kb = EnhancedKnowledgeBase()
    re = ReasoningEngine(kb)

    # Adding sample data
    kb.add_entity("Cat", {"type": "animal", "legs": 4, "fur": True}, entity_type="Animal", certainty=1.0)
    kb.add_entity("Dog", {"type": "animal", "legs": 4, "fur": True}, entity_type="Animal", certainty=1.0)
    kb.add_entity("Bird", {"type": "animal", "legs": 2, "feathers": True}, entity_type="Animal", certainty=1.0)
    kb.add_entity("Fish", {"type": "animal", "fins": True, "gills": True}, entity_type="Animal", certainty=1.0)

    kb.add_relationship("Cat", "Dog", "similar_to", {"reason": "both mammals"}, certainty=0.9)
    kb.add_relationship("Dog", "Bird", "different_from", {"reason": "different number of legs"}, certainty=0.8)

    # Test the reasoning engine
    print("Transitive relationships:")
    print(re.infer_transitive_relationships("Cat", "similar_to"))

    print("\nCommon attributes between Cat and Dog:")
    print(re.find_common_attributes("Cat", "Dog"))

    print("\nSuggested new relationships for Cat:")
    print(re.suggest_new_relationships("Cat"))

    print("\nHypothesis for Fish:")
    print(re.generate_hypothesis("Fish"))

    print("\nExplanation of inference:")
    kb.add_entity("Lion", {"type": "animal"}, entity_type="Animal", certainty=1.0)
    kb.add_relationship("Lion", "Cat", "similar_to", {"reason": "both felines"}, certainty=0.9)
    print(re.explain_inference("Lion", "fur"))
