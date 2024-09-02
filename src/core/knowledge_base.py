# BeSAI/src/core/knowledge_base.py

import networkx as nx
from typing import Any, List, Dict, Tuple, Optional

class KnowledgeBase:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_entity(self, entity: str, attributes: Dict[str, Any] = None):
        """
        Add an entity to the knowledge base.
        
        :param entity: The name of the entity
        :param attributes: A dictionary of attributes for the entity
        """
        if attributes is None:
            attributes = {}
        self.graph.add_node(entity, **attributes)

    def add_relationship(self, entity1: str, entity2: str, relationship: str, attributes: Dict[str, Any] = None):
        """
        Add a relationship between two entities.
        
        :param entity1: The name of the first entity
        :param entity2: The name of the second entity
        :param relationship: The type of relationship
        :param attributes: A dictionary of attributes for the relationship
        """
        if attributes is None:
            attributes = {}
        self.graph.add_edge(entity1, entity2, relationship=relationship, **attributes)

    def get_entity(self, entity: str) -> Optional[Dict[str, Any]]:
        """
        Get the attributes of an entity.
        
        :param entity: The name of the entity
        :return: A dictionary of entity attributes or None if the entity doesn't exist
        """
        return dict(self.graph.nodes[entity]) if entity in self.graph.nodes else None

    def get_relationships(self, entity: str) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Get all relationships of an entity.
        
        :param entity: The name of the entity
        :return: A list of tuples (related_entity, relationship_type, attributes)
        """
        if entity not in self.graph.nodes:
            return []
        relationships = []
        for _, related_entity, data in self.graph.edges(entity, data=True):
            relationships.append((related_entity, data['relationship'], data))
        return relationships

    def query(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform a simple query on the knowledge base.
        
        :param query: A string query (for now, just match entity names)
        :return: A list of matching entities with their attributes
        """
        return [
            {'entity': node, 'attributes': dict(data)}
            for node, data in self.graph.nodes(data=True)
            if query.lower() in node.lower()
        ]

    def update_entity(self, entity: str, attributes: Dict[str, Any]):
        """
        Update the attributes of an entity.
        
        :param entity: The name of the entity
        :param attributes: A dictionary of attributes to update
        """
        self.graph.nodes[entity].update(attributes)

    def remove_entity(self, entity: str):
        """
        Remove an entity and all its relationships from the knowledge base.
        
        :param entity: The name of the entity to remove
        """
        self.graph.remove_node(entity)

    def get_all_entities(self) -> List[str]:
        """
        Get a list of all entities in the knowledge base.
        
        :return: A list of entity names
        """
        return list(self.graph.nodes())

    def get_entity_count(self) -> int:
        """
        Get the total number of entities in the knowledge base.
        
        :return: The number of entities
        """
        return self.graph.number_of_nodes()

    def get_relationship_count(self) -> int:
        """
        Get the total number of relationships in the knowledge base.
        
        :return: The number of relationships
        """
        return self.graph.number_of_edges()

# Example usage
if __name__ == "__main__":
    kb = KnowledgeBase()

    # Adding entities
    kb.add_entity("Python", {"type": "programming_language", "paradigm": "multi-paradigm"})
    kb.add_entity("Java", {"type": "programming_language", "paradigm": "object-oriented"})
    kb.add_entity("AI", {"type": "field", "description": "Artificial Intelligence"})

    # Adding relationships
    kb.add_relationship("Python", "AI", "used_in", {"strength": "high"})
    kb.add_relationship("Java", "AI", "used_in", {"strength": "medium"})

    # Querying
    results = kb.query("python")
    print("Query results for 'python':", results)

    # Getting relationships
    python_relationships = kb.get_relationships("Python")
    print("Python relationships:", python_relationships)

    # Updating an entity
    kb.update_entity("Python", {"version": "3.9"})
    print("Updated Python entity:", kb.get_entity("Python"))

    # Print some statistics
    print("Total entities:", kb.get_entity_count())
    print("Total relationships:", kb.get_relationship_count())
