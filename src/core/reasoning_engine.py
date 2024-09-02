# BeSAI/src/core/reasoning_engine.py

from typing import List, Dict, Any, Optional
from .knowledge_base import KnowledgeBase

class ReasoningEngine:
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base

    def infer_transitive_relationships(self, start_entity: str, relationship_type: str) -> List[Dict[str, Any]]:
        """
        Infer transitive relationships from a starting entity.
        
        :param start_entity: The entity to start the inference from
        :param relationship_type: The type of relationship to follow
        :return: A list of inferred relationships
        """
        visited = set()
        to_visit = [start_entity]
        inferred_relationships = []

        while to_visit:
            current_entity = to_visit.pop(0)
            if current_entity in visited:
                continue
            visited.add(current_entity)

            relationships = self.kb.get_relationships(current_entity)
            for related_entity, rel_type, _ in relationships:
                if rel_type == relationship_type:
                    inferred_relationships.append({
                        "from": start_entity,
                        "to": related_entity,
                        "relationship": relationship_type,
                        "inferred": current_entity != start_entity
                    })
                    to_visit.append(related_entity)

        return inferred_relationships

    def find_common_attributes(self, entity1: str, entity2: str) -> Dict[str, Any]:
        """
        Find common attributes between two entities.
        
        :param entity1: The first entity
        :param entity2: The second entity
        :return: A dictionary of common attributes
        """
        attrs1 = self.kb.get_entity(entity1)
        attrs2 = self.kb.get_entity(entity2)
        
        common_attrs = {}
        for key, value in attrs1.items():
            if key in attrs2 and attrs2[key] == value:
                common_attrs[key] = value
        
        return common_attrs

    def suggest_new_relationships(self, entity: str) -> List[Dict[str, Any]]:
        """
        Suggest new relationships based on common attributes with other entities.
        
        :param entity: The entity to suggest relationships for
        :return: A list of suggested relationships
        """
        entity_attrs = self.kb.get_entity(entity)
        all_entities = self.kb.get_all_entities()
        suggestions = []

        for other_entity in all_entities:
            if other_entity != entity:
                common_attrs = self.find_common_attributes(entity, other_entity)
                if common_attrs:
                    suggestions.append({
                        "entity1": entity,
                        "entity2": other_entity,
                        "common_attributes": common_attrs,
                        "suggested_relationship": "related_to"
                    })

        return suggestions

    def infer_attribute_from_relationships(self, entity: str, attribute: str) -> Any:
        """
        Infer an attribute value for an entity based on its relationships.
        
        :param entity: The entity to infer the attribute for
        :param attribute: The attribute to infer
        :return: The inferred attribute value, or None if it can't be inferred
        """
        relationships = self.kb.get_relationships(entity)
        for related_entity, _, _ in relationships:
            related_attrs = self.kb.get_entity(related_entity)
            if attribute in related_attrs:
                return related_attrs[attribute]
        return None

    def generate_hypothesis(self, entity: str) -> Optional[Dict[str, Any]]:
        """
        Generate a hypothesis about an entity based on its relationships and attributes.
        
        :param entity: The entity to generate a hypothesis for
        :return: A dictionary containing the generated hypothesis, or None if the entity doesn't exist
        """
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
                        hypothesis["inferred_attributes"][attr] = value

        # Suggest potential relationships
        hypothesis["potential_relationships"] = self.suggest_new_relationships(entity)

        return hypothesis

# Example usage
if __name__ == "__main__":
    kb = KnowledgeBase()
    kb.add_entity("Cat", {"type": "animal", "legs": 4, "fur": True})
    kb.add_entity("Dog", {"type": "animal", "legs": 4, "fur": True})
    kb.add_entity("Bird", {"type": "animal", "legs": 2, "feathers": True})
    kb.add_entity("Fish", {"type": "animal", "fins": True, "gills": True})

    kb.add_relationship("Cat", "Dog", "similar_to", {"reason": "both mammals"})
    kb.add_relationship("Dog", "Bird", "different_from", {"reason": "different number of legs"})

    re = ReasoningEngine(kb)

    print("Transitive relationships:")
    print(re.infer_transitive_relationships("Cat", "similar_to"))

    print("\nCommon attributes between Cat and Dog:")
    print(re.find_common_attributes("Cat", "Dog"))

    print("\nSuggested new relationships for Cat:")
    print(re.suggest_new_relationships("Cat"))

    print("\nHypothesis for Fish:")
    print(re.generate_hypothesis("Fish"))
