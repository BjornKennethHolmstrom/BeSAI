import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
from typing import Any, List, Dict, Tuple, Optional
from collections import defaultdict
from collections import Counter

class EnhancedKnowledgeBase:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.entity_types = {}

    def add_entity(self, entity: str, attributes: Dict[str, Any] = None, entity_type: str = None, certainty: float = 1.0):
        if attributes is None:
            attributes = {}
        certainty_dict = {attr: certainty for attr in attributes}
        attributes['certainty'] = certainty_dict
        self.graph.add_node(entity, **attributes)
        if entity_type:
            self.entity_types[entity_type].add(entity)

    def add_relationship(self, entity1: str, entity2: str, relationship: str, attributes: Dict[str, Any] = None, certainty: float = 1.0):
        if attributes is None:
            attributes = {}
        attributes['certainty'] = certainty
        self.graph.add_edge(entity1, entity2, key=relationship, **attributes)

    def get_entity(self, entity: str) -> Dict[str, Any]:
        # This method should return the entity's data
        # If it doesn't exist in the knowledge base, return an empty dict
        return self.graph.nodes.get(entity, {})

    def get_relationships(self, entity: str) -> List[Tuple[str, str, Dict[str, Any]]]:
            if entity not in self.graph.nodes:
                return []
            relationships = []
            for _, related_entity, key, data in self.graph.edges(entity, data=True, keys=True):
                relationships.append((related_entity, key, data))
            return relationships

    def query(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        results = []
        for node, data in self.graph.nodes(data=True):
            if all(key in data and data[key] == value for key, value in query.items()):
                results.append({'entity': node, 'attributes': data})
        return results

    def update_entity(self, entity: str, attributes: Dict[str, Any], certainty: float = 1.0):
        if entity in self.graph.nodes:
            current_attrs = self.graph.nodes[entity]
            current_certainty = current_attrs.get('certainty', {})
            for key, value in attributes.items():
                current_attrs[key] = value
                if isinstance(current_certainty, dict):
                    current_certainty[key] = (current_certainty.get(key, 1.0) + certainty) / 2
                else:
                    current_certainty = {key: (1.0 + certainty) / 2}
            current_attrs['certainty'] = current_certainty

    def get_entities_by_type(self, entity_type: str) -> List[str]:
        return list(self.entity_types.get(entity_type, []))

    def get_entity_types(self) -> List[str]:
        return list(self.entity_types.keys())

    def get_all_entities(self) -> List[str]:
        """
        Get a list of all entities in the knowledge base.
        
        :return: A list of entity names
        """
        return list(self.graph.nodes())

    def find_path(self, start_entity: str, end_entity: str) -> List[Tuple[str, str, str]]:
        try:
            path = nx.shortest_path(self.graph, start_entity, end_entity)
            result = []
            for i in range(len(path) - 1):
                edge_data = self.graph.get_edge_data(path[i], path[i+1])
                if edge_data:
                    # Get the first edge if there are multiple edges
                    first_edge = next(iter(edge_data.values()))
                    relationship = first_edge.get('relationship', 'related_to')
                else:
                    relationship = 'related_to'
                result.append((path[i], relationship, path[i+1]))
            return result
        except nx.NetworkXNoPath:
            return []

    def get_subgraph(self, entities: List[str]) -> 'EnhancedKnowledgeBase':
        subgraph = self.graph.subgraph(entities)
        new_kb = EnhancedKnowledgeBase()
        new_kb.graph = subgraph
        for entity_type, entity_set in self.entity_types.items():
            new_kb.entity_types[entity_type] = entity_set.intersection(set(entities))
        return new_kb

    def calculate_relevance(self, entity: str) -> float:
        try:
            relationships = self.get_relationships(entity)
            if not relationships:
                return 0.0

            # Count the number of relationships for each related entity
            related_entity_counts = Counter(rel[0] for rel in relationships if isinstance(rel, tuple) and len(rel) > 0)

            if not related_entity_counts:
                return 0.0

            # Calculate the average number of relationships
            avg_relationships = sum(related_entity_counts.values()) / len(related_entity_counts)

            # Calculate relevance based on the number and strength of relationships
            relevance = 0.0
            for rel, count in related_entity_counts.items():
                entity_data = self.get_entity(rel)
                if isinstance(entity_data, dict):
                    certainty = entity_data.get('certainty', 1.0)
                    if isinstance(certainty, dict):
                        # If certainty is a dict, use the average of its values
                        certainty = sum(certainty.values()) / len(certainty) if certainty else 1.0
                    relevance += certainty * count
                else:
                    relevance += count  # Default to using just the count if entity_data is not a dict

            relevance /= len(relationships) if relationships else 1

            # Adjust relevance based on the average number of relationships
            adjusted_relevance = relevance * (1 + avg_relationships / 10)

            return min(1.0, adjusted_relevance)  # Ensure relevance is between 0 and 1
        except Exception as e:
            print(f"Error calculating relevance for {entity}: {str(e)}")
            return 0.0  # Return 0 relevance if there's an error

    def remove_entity(self, entity: str):
        """
        Remove an entity and all its relationships from the knowledge base.
        
        :param entity: The name of the entity to remove
        """
        if entity in self.graph:
            self.graph.remove_node(entity)
            # Remove the entity from entity_types
            for entity_type, entities in self.entity_types.items():
                if entity in entities:
                    entities.remove(entity)

# Example usage
if __name__ == "__main__":
    kb = EnhancedKnowledgeBase()

    # Adding entities with types and certainty
    kb.add_entity("Python", {"type": "programming_language", "paradigm": "multi-paradigm"}, entity_type="Language", certainty=1.0)
    kb.add_entity("Java", {"type": "programming_language", "paradigm": "object-oriented"}, entity_type="Language", certainty=1.0)
    kb.add_entity("AI", {"type": "field", "description": "Artificial Intelligence"}, entity_type="Field", certainty=1.0)
    kb.add_entity("Machine Learning", {"type": "subfield", "description": "Subset of AI"}, entity_type="Field", certainty=0.9)

    # Adding relationships with certainty
    kb.add_relationship("Python", "AI", "used_in", {"strength": "high"}, certainty=0.8)
    kb.add_relationship("Java", "AI", "used_in", {"strength": "medium"}, certainty=0.7)
    kb.add_relationship("AI", "Machine Learning", "includes", certainty=1.0)

    # Querying
    results = kb.query({"type": "programming_language"})
    print("Query results for programming languages:", results)

    # Getting relationships
    python_relationships = kb.get_relationships("Python")
    print("Python relationships:", python_relationships)

    # Updating an entity
    kb.update_entity("Python", {"version": "3.9"}, certainty=0.9)
    print("Updated Python entity:", kb.get_entity("Python"))

    # Getting entities by type
    languages = kb.get_entities_by_type("Language")
    print("Languages:", languages)

    # Finding path
    path = kb.find_path("Python", "Machine Learning")
    print("Path from Python to Machine Learning:", path)

    # Finding a non-existent path
    no_path = kb.find_path("Python", "Non-existent Entity")
    print("Path to non-existent entity:", no_path)

    # Getting subgraph
    subgraph = kb.get_subgraph(["Python", "AI", "Machine Learning"])
    print("Subgraph entities:", subgraph.graph.nodes())

    # Print some statistics
    print("Total entities:", kb.graph.number_of_nodes())
    print("Total relationships:", kb.graph.number_of_edges())
    print("Entity types:", kb.get_entity_types())
