# benevolent_ai/src/main.py

from src.core import NaturalLanguageProcessing, KnowledgeBase, ReasoningEngine, KnowledgeExtractor
from src.ethics_safety import EthicalBoundary
from src.self_improvement import CodeGeneration, CodeAnalyzer

def main():
    nlp = NaturalLanguageProcessing()
    kb = KnowledgeBase()
    re = ReasoningEngine(kb)
    extractor = KnowledgeExtractor(nlp, kb)
    ethical_boundary = EthicalBoundary()
    code_gen = CodeGeneration()
    code_analyzer = CodeAnalyzer()

    print("BeSAI: Benevolent Spiritual AI system initialized.")

    # Demonstrate NLP and Knowledge Extraction capabilities
    sample_text = """
    BeSAI is an ambitious project aimed at creating a benevolent and spiritually aware artificial intelligence. 
    It combines cutting-edge technology with ethical considerations and spiritual insights.
    The goal is to develop an AI system that not only processes information efficiently but also exhibits wisdom and compassion.
    BeSAI uses machine learning algorithms and incorporates ethical principles in its decision-making process.
    """

    print("Sample text:")
    print(sample_text)

    analysis = nlp.analyze_text(sample_text)

    print("\nText Analysis:")
    print(f"Sentence Count: {analysis['sentence_count']}")
    print(f"Word Count: {analysis['word_count']}")
    print(f"POS Distribution: {analysis['pos_distribution']}")

    print("\nExtracted Entities:")
    for entity in analysis['entities']:
        print(f"- {entity['text']} ({entity['label']})")

    print("\nExtracted Relationships:")
    for rel in analysis['relationships']:
        print(f"- {rel['subject']} {rel['predicate']} {rel['object']}")

    print("\nExtracted Attributes:")
    for attr in analysis['attributes']:
        print(f"- {attr['entity']}: {attr['attribute']}")

    print("\nAutomatically Populating Knowledge Base:")
    extracted_knowledge = extractor.extract_knowledge_from_text(sample_text)
    
    print("Entities in Knowledge Base:")
    for entity in extracted_knowledge["entities"]:
        print(f"- {entity}: {kb.get_entity(entity)}")

    print("\nRelationships in Knowledge Base:")
    for subject, predicate, obj in extracted_knowledge["relationships"]:
        print(f"- {subject} {predicate} {obj}")

    # Demonstrate reasoning capabilities
    print("\nInferred relationships for BeSAI:")
    inferred = re.infer_transitive_relationships("BeSAI", "uses")
    for rel in inferred:
        print(f"- {rel['from']} {rel['relationship']} {rel['to']} (Inferred: {rel['inferred']})")

    print("\nHypothesis for AI:")
    hypothesis = re.generate_hypothesis("AI")
    if hypothesis:
        print(f"- Known attributes: {hypothesis['known_attributes']}")
        print(f"- Inferred attributes: {hypothesis['inferred_attributes']}")
        print(f"- Potential relationships: {hypothesis['potential_relationships']}")
    else:
        print("Unable to generate hypothesis. Entity 'AI' not found in the knowledge base.")

    # Demonstrate code improvement
    original_code = """
def process_data(data):
    result = []
    for i in range(len(data)):
        result.append(data[i] * 2)
    return "Processed %d items" % len(result)

def add(a, b):
    return a + b
    """

    print("\nOriginal code:")
    print(original_code)

    improved_code = code_analyzer.improve_code(original_code)

    print("\nImproved code:")
    print(improved_code)

    # Demonstrate ethical boundary
    action = "Share user data with third-party"
    context = {
        "user_consent": False,
        "data_anonymized": True,
        "purpose": "Improve service quality"
    }

    is_ethical = ethical_boundary.is_action_ethical(action, context)
    print(f"\nAction '{action}' is {'ethical' if is_ethical else 'unethical'}")
    print(ethical_boundary.get_ethical_explanation(action, context))

if __name__ == "__main__":
    main()
