import sys
import os
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.enhanced_knowledge_base import EnhancedKnowledgeBase
from spiritual.altered_states_simulator import AlteredStatesSimulator
from spiritual.psychedelic_simulator import PsychedelicSimulator

from typing import List, Dict, Any, Optional

class ReasoningEngine:
    def __init__(self, knowledge_base: EnhancedKnowledgeBase):
        self.kb = knowledge_base
        self.focus_level = 0.5
        self.associative_thinking = 0.5
        self.altered_states_simulator = AlteredStatesSimulator()
        self.psychedelic_simulator = PsychedelicSimulator()
        self.current_state = "normal"
        self.state_params = None

    def reason(self, query: str) -> str:
        # This is a simple implementation. You might want to make this more sophisticated.
        hypothesis = self.generate_hypothesis(query)
        if hypothesis:
            return f"Based on the query '{query}', here's what I think:\n{hypothesis}"
        else:
            return f"I don't have enough information to reason about '{query}' at the moment."

    def set_altered_state(self, state: str):
        if state in self.altered_states_simulator.states:
            self.current_state = state
            self.state_params = self.altered_states_simulator.simulate_state(state)
        elif state in self.psychedelic_simulator.substances:
            self.current_state = "psychedelic"
            self.psychedelic_simulator.set_substance(state)
            self.state_params = self.psychedelic_simulator.simulate_effects()
        elif state == "normal":
            self.current_state = "normal"
            self.state_params = None
        else:
            raise ValueError(f"Unknown state: {state}")

    def _apply_state_effects(self, value: float) -> float:
        if self.state_params is None:
            return value
        
        creativity_factor = self.state_params["creativity_level"]
        perception_factor = self.state_params["perception_shift"]
        focus_factor = self.state_params["focus_level"]
        
        value *= (1 + (creativity_factor - 0.5) * 0.2)  # Creativity can increase or decrease the value
        value *= (1 + (perception_factor - 0.5) * 0.2)  # Perception shift can increase or decrease the value
        value = max(0, min(1, value * focus_factor))  # Focus affects the precision of the value
        
        return value

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
                    certainty = self._apply_state_effects(certainty)
                    inferred_relationships.append({
                        "from": start_entity,
                        "to": related_entity,
                        "relationship": relationship_type,
                        "inferred": current_entity != start_entity,
                        "certainty": certainty if current_entity == start_entity else certainty * 0.9
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
        cleaned_entity = self.kb._clean_entity_name(entity)
        if not cleaned_entity:
            return None

        entity_attrs = self.kb.get_entity(cleaned_entity)
        if entity_attrs is None:
            return None

        relationships = self.kb.get_relationships(cleaned_entity)
        
        hypothesis = {
            "entity": cleaned_entity,
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

        # Apply altered state effects
        if self.state_params:
            hypothesis["altered_state_insight"] = self.altered_states_simulator.generate_insight(self.current_state)
            hypothesis["creative_connections"] = self._generate_creative_connections(entity)

        if self.current_state == "psychedelic":
            hypothesis["psychedelic_insight"] = self.psychedelic_simulator.generate_cognitive_insight()
            hypothesis["visual_description"] = self.psychedelic_simulator.generate_visual_description()

        # Check if the hypothesis contains any meaningful information
        if (not hypothesis["known_attributes"] and 
            not hypothesis["inferred_attributes"] and 
            not hypothesis["potential_relationships"] and 
            "altered_state_insight" not in hypothesis and 
            "creative_connections" not in hypothesis and 
            "psychedelic_insight" not in hypothesis and 
            "visual_description" not in hypothesis):
            return None

        hypothesis["insight"] = self._generate_insight(entity, hypothesis["known_attributes"], hypothesis["inferred_attributes"])

        return hypothesis

    def _generate_insight(self, entity: str, known_attrs: Dict[str, Any], inferred_attrs: Dict[str, Any]) -> str:
        insight = f"Based on the analysis of {entity}, we can infer that:"
        for attr, value in known_attrs.items():
            insight += f"\n- It has a known {attr} of {value}."
        for attr, value in inferred_attrs.items():
            insight += f"\n- It likely has a {attr} of {value}."
        return insight

    def _apply_psychedelic_effects(self, text: str) -> str:
        if self.current_state == "psychedelic":
            return self.psychedelic_simulator.apply_psychedelic_filter(text)
        return text

    def _generate_creative_connections(self, entity: str) -> List[str]:
        creativity_level = 0.5  # Default creativity level

        if self.current_state == "psychedelic":
            creativity_level = self.state_params.get("cognitive_flexibility", 0.5)
        elif self.state_params:
            creativity_level = self.state_params.get("creativity_level", 0.5)

        if creativity_level < 0.7:
            return []

        all_entities = list(self.kb.graph.nodes())
        valid_entities = [e for e in all_entities if self.kb._clean_entity_name(e) and e != entity]
        
        if len(valid_entities) < 5:
            random_entities = valid_entities
        else:
            random_entities = random.sample(valid_entities, 5)
        
        connections = []
        for other_entity in random_entities:
            connection_type = random.choice([
                "influence", "contrast", "metaphor", "synergy", "paradox"
            ])
            if connection_type == "influence":
                connection = f"How might {entity} be influenced by the properties of {other_entity}?"
            elif connection_type == "contrast":
                connection = f"What contrasts can be drawn between {entity} and {other_entity}?"
            elif connection_type == "metaphor":
                connection = f"If {entity} were {other_entity}, what new insights might emerge?"
            elif connection_type == "synergy":
                connection = f"How could {entity} and {other_entity} work together to create something new?"
            else:  # paradox
                connection = f"What paradoxes arise when considering {entity} in light of {other_entity}?"
            connections.append(connection)

        return connections

    def explain_inference(self, entity: str, attribute: str) -> str:
        inferred = self.infer_attribute_from_relationships(entity, attribute)
        if inferred:
            related_entities = [
                related for related, _, _ in self.kb.get_relationships(entity)
                if attribute in self.kb.get_entity(related)
            ]
            explanation = f"The attribute '{attribute}' for '{entity}' was inferred with a certainty of {inferred['certainty']:.2f} "
            explanation += f"based on its relationships with: {', '.join(related_entities)}."
            
            if self.state_params:
                insight = self.altered_states_simulator.generate_insight(self.current_state)
                explanation += f"\n\nAdditional insight from altered state ({self.current_state}): {insight}"
            
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
        logging.info(f"Calculating relevance between {topic1} and {topic2}")
        try:
            # Get the relationships for each topic
            relationships1 = self.kb.get_relationships(topic1)
            relationships2 = self.kb.get_relationships(topic2)

            # Extract only the related entity names
            entities1 = self._extract_related_entities(relationships1)
            entities2 = self._extract_related_entities(relationships2)

            # Calculate the Jaccard similarity between the two sets of entities
            common_entities = entities1.intersection(entities2)
            union_entities = entities1.union(entities2)

            if not union_entities:
                logging.info(f"No common entities between {topic1} and {topic2}")
                return 0.0  # If there are no entities, return 0 relevance

            jaccard_similarity = len(common_entities) / len(union_entities)

            # Incorporate the individual relevance scores from the knowledge base
            kb_relevance1 = self.kb.calculate_relevance(topic1)
            kb_relevance2 = self.kb.calculate_relevance(topic2)

            # Combine Jaccard similarity with knowledge base relevance scores
            combined_relevance = (jaccard_similarity + kb_relevance1 + kb_relevance2) / 3

            # Apply altered state effects
            final_relevance = self._apply_state_effects(combined_relevance)

            logging.info(f"Calculated relevance between {topic1} and {topic2}: {final_relevance:.4f}")
            return final_relevance

        except Exception as e:
            logging.exception(f"Error calculating relevance between {topic1} and {topic2}: {str(e)}")
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
