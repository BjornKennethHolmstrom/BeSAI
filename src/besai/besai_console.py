import os
from logging.handlers import RotatingFileHandler
import threading
import cmd
import schedule
import time
import traceback
import signal
import importlib
from besai.core.enhanced_knowledge_base import EnhancedKnowledgeBase
from besai.core.reasoning_engine import ReasoningEngine
from besai.core.enhanced_natural_language_processing import EnhancedNaturalLanguageProcessing
from besai.core.learning_system import LearningSystem
from besai.core.autonomous_learning import AutonomousLearning
from besai.core.meta_cognition import Metacognition
from besai.spiritual.altered_states_simulator import AlteredStatesSimulator
from besai.spiritual.curiosity_engine import CuriosityEngine
from besai.spiritual.self_reflection import SelfReflection
from besai.spiritual.self_concept import SelfConcept
import personality_module
importlib.reload(personality_module)
from besai.personality_module import PersonalityModule

import logging
from besai.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class BeSAIConsole(cmd.Cmd):
    intro = "Welcome to the BeSAI console. Type 'help' for a list of commands."
    prompt = "BeSAI> "

    def __init__(self):
        super().__init__()
        self.kb = EnhancedKnowledgeBase()
        self.re = ReasoningEngine(self.kb)
        self.nlp = EnhancedNaturalLanguageProcessing(self.kb, self.re)
        
        # Create Metacognition object first
        self.metacognition = Metacognition(self.kb, None, self.re)  # We'll set the learning_system later
        
        # Now create LearningSystem with Metacognition
        self.ls = LearningSystem(self.kb, self.nlp, self.re, self.metacognition)
        
        # Set the learning_system in Metacognition
        self.metacognition.set_learning_system(self.ls)
        
        # Finally, create AutonomousLearning with all components
        self.al = AutonomousLearning(self.kb, self.nlp, self.ls, self.re, self.metacognition)
        
        self.al.load_knowledge()

        # Some extra bonuses
        self.altered_states_simulator = AlteredStatesSimulator()

        # Initialize PersonalityModule
        self.personality = PersonalityModule()
        self._load_personality_data()

        # Initialize CuriosityEngine
        self.curiosity_engine = CuriosityEngine(self.al)

        self.self_concept = SelfConcept(self.kb, self.re, self.metacognition, self.personality)
        self.self_reflection = SelfReflection(self.kb, self.metacognition, self.re, self.self_concept)

        self.current_state = "normal"

        # Set up signal handler for graceful exit
        self.shutting_down = False
        signal.signal(signal.SIGINT, self.signal_handler)

    def _load_personality_data(self):
        file_paths = [
            os.path.join(".", "docs/inheritance-sources", "bkh-source-blog-posts.md"),
            os.path.join(".", "docs/inheritance-sources", "bkh-source-novel-excerpt.md"),
            os.path.join(".", "docs/inheritance-sources", "bkh-source-poems.md"),
            os.path.join(".", "docs/inheritance-sources", "bkh-source-home-pages.md"),
            os.path.join(".", "docs/inheritance-sources", "personality-traits.md")
        ]
        self.personality.load_text_samples_from_files(file_paths)
        if not self.personality.learned_phrases:
            logging.warning("No personality data was loaded. The system will use default responses.")

    def default(self, line):
        try:
            personality_response = self.personality.generate_response(line)
            reasoning_response = self.re.reason(line)
            combined_response = f"Personality: {personality_response}\nReasoning: {reasoning_response}"
            return combined_response
        except Exception as e:
            logging.error(f"An error occurred while processing the input: {line}")
            logging.error(f"Error details: {str(e)}")
            logging.error(traceback.format_exc())
            return f"Error: {str(e)}"

    def do_explore(self, arg):
        logger.info(f"do_explore called with arg: {arg}")
        if not arg:
            arg = self.curiosity_engine.suggest_exploration()
        logger.info(f"Exploring topic: {arg}")
        result = self.al.explore_topic(arg)
        logger.info(f"Exploration result: {result}")
        return f"Explored topic: {arg}. Result: {result}"

    def do_insight(self, arg):
        logger.info(f"do_insight called with arg: {arg}")
        if not arg:
            return "Please provide a topic to get insight on."
        logger.info(f"Generating insight for: {arg}")
        insight = self.al.get_insight(arg)
        if "No information available" in insight:
            logger.info(f"No information available for the topic: {arg}. Exploring...")
            exploration_result = self.al.explore_topic(arg)
            logger.info(f"Exploration result: {exploration_result}")
            insight = self.al.get_insight(arg)
        logger.info(f"Insight: {insight}")
        return insight

    def do_reason(self, arg):
        logger.info(f"do_reason called with arg: {arg}")
        if not arg:
            return "Please provide a query for reasoning."
        logger.info(f"Reasoning about: {arg}")
        result = self.re.generate_hypothesis(arg)
        if not result:
            logger.info(f"No information available for reasoning about: {arg}. Exploring...")
            self.do_explore(arg)
            result = self.re.generate_hypothesis(arg)
        logger.info(f"Reasoning result: {result}")
        if result:
            output = f"Reasoning about: {arg}\n"
            output += f"Entity: {result['entity']}\n"
            if result['known_attributes']:
                output += "Known attributes:\n"
                for key, value in result['known_attributes'].items():
                    output += f"  - {key}: {value}\n"
            if result['inferred_attributes']:
                output += "Inferred attributes:\n"
                for key, value in result['inferred_attributes'].items():
                    output += f"  - {key}: {value}\n"
            if result['potential_relationships']:
                output += "Potential relationships:\n"
                for rel in result['potential_relationships']:
                    output += f"  - {rel['entity1']} {rel['suggested_relationship']} {rel['entity2']} (certainty: {rel['certainty']:.2f})\n"
            return output
        else:
            return f"Unable to generate a hypothesis for the query: {arg} even after exploration."

    def do_clean_kb(self, arg):
        """Clean the knowledge base by removing or merging invalid entities"""
        try:
            initial_entity_count = len(self.kb.get_all_entities())
            self.kb.clean_entities()
            final_entity_count = len(self.kb.get_all_entities())
            removed_count = initial_entity_count - final_entity_count
            print(f"Knowledge base cleaned. Removed or merged {removed_count} entities.")
            print(f"Current entity count: {final_entity_count}")
        except Exception as e:
            logging.error(f"Error cleaning knowledge base: {str(e)}")
            logging.error(traceback.format_exc())

    def do_meta_assess(self, arg):
        """Perform a metacognitive assessment of a topic: meta_assess TOPIC"""
        if not arg:
            print("Please provide a topic for metacognitive assessment.")
            return
        try:
            assessment = self.al.metacognition.assess_knowledge(arg)
            print(f"Metacognitive Assessment for {arg}:")
            print(f"Knowledge Depth: {assessment['knowledge_depth']}")
            print(f"Relevance: {assessment['relevance']:.2f}")
            print(f"Confidence: {assessment['confidence']:.2f}")
            if assessment['gaps']:
                print(f"Identified Knowledge Gaps: {', '.join(assessment['gaps'])}")
            else:
                print("No significant knowledge gaps identified.")
        except Exception as e:
            logging.error(f"Error performing metacognitive assessment: {str(e)}")
            logging.error(traceback.format_exc())

    def do_learning_progress(self, arg):
        """View the learning progress for a topic: learning_progress TOPIC"""
        if not arg:
            print("Please provide a topic to view learning progress.")
            return
        try:
            progress = self.al.metacognition.analyze_learning_process(arg)
            print(f"Learning Progress for {arg}:")
            print(f"Learning Rate: {progress['learning_rate']:.2f}")
            print(f"Source Diversity: {progress['source_diversity']:.2f}")
            print(f"Understanding Depth: {progress['understanding_depth']:.2f}")
        except Exception as e:
            logging.error(f"Error retrieving learning progress: {str(e)}")
            logging.error(traceback.format_exc())

    def do_meta_insight(self, arg):
        """Generate a meta-insight for a topic: meta_insight TOPIC"""
        if not arg:
            print("Please provide a topic for meta-insight generation.")
            return
        try:
            insight = self.al.metacognition.generate_meta_insight(arg)
            print(insight)
        except Exception as e:
            logging.error(f"Error generating meta-insight: {str(e)}")
            logging.error(traceback.format_exc())

    def do_bias_analysis(self, arg):
        """Perform a bias analysis for a topic: bias_analysis TOPIC"""
        if not arg:
            print("Please provide a topic for bias analysis.")
            return
        try:
            biases = self.al.metacognition.detect_biases(arg)
            acknowledgment = self.al.metacognition.acknowledge_biases(arg)
            
            print(f"Bias Analysis for {arg}:")
            print(f"Source Bias: {biases['source_bias']:.2f}")
            print(f"Sentiment Bias: {biases['sentiment_bias']:.2f}")
            print(f"Perspective Bias: {biases['perspective_bias']:.2f}")
            print("\nBias Acknowledgment:")
            print(acknowledgment)
        except Exception as e:
            logging.error(f"Error performing bias analysis: {str(e)}")
            logging.error(traceback.format_exc())

    def do_generate_curiosity(self, arg):
        """Generate a curiosity prompt"""
        try:
            prompt = self.curiosity_engine.generate_curiosity_prompt()
            print(f"Curiosity prompt generated:")
            print(f"Topic: {prompt['topic']}")
            print(f"Question: {prompt['question']}")
            print(f"Strategy: {prompt['strategy']}")
        except Exception as e:
            logging.error(f"Error generating curiosity prompt: {str(e)}")
            print(f"An error occurred while generating the curiosity prompt: {str(e)}")


    def do_process_learning(self, arg):
        """Process learning outcome: process_learning TOPIC "NEW_INFORMATION" """
        args = arg.split('" ', 1)
        if len(args) != 2:
            print("Invalid format. Use: process_learning TOPIC \"NEW_INFORMATION\"")
            return
        topic = args[0].strip()
        new_information = args[1].strip('"')
        self.curiosity_engine.process_learning_outcome(topic, new_information)
        print(f"Learning outcome processed for topic: {topic}")

    def do_auto_explore(self, arg):
        """Start an automatic exploration cycle based on BeSAI's curiosity"""
        try:
            topic = self.curiosity_engine.suggest_exploration()
            self.do_explore(topic)
        except Exception as e:
            logging.error(f"Error during auto-exploration: {str(e)}")
            print(f"An error occurred during auto-exploration: {str(e)}")
            print("Please try again later.")

    def do_curiosity_metrics(self, arg):
        """Show curiosity metrics"""
        try:
            metrics = self.curiosity_engine.get_curiosity_metrics()
            print("Curiosity Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
        except Exception as e:
            logging.error(f"Error retrieving curiosity metrics: {str(e)}")
            print(f"An error occurred while retrieving curiosity metrics: {str(e)}")

    def do_set_state(self, arg):
        """Set the current altered state: set_state STATE"""
        try:
            self.re.set_altered_state(arg)
            print(f"Current state set to: {arg}")
            if arg in self.re.psychedelic_simulator.substances:
                print(self.re.psychedelic_simulator.generate_experience_report())
            elif arg != "normal":
                description = self.altered_states_simulator.get_state_description(arg)
                print(description)
        except ValueError as e:
            print(f"Error: {str(e)}")
            print("Available states:", ", ".join(list(self.altered_states_simulator.states.keys()) + 
                                                 list(self.re.psychedelic_simulator.substances.keys()) + 
                                                 ["normal"]))

    def do_simulate_state(self, arg):
        """Simulate an altered state of consciousness: simulate_state STATE"""
        if not arg:
            print("Error: Please specify a state to simulate.")
            print("Available states:", ", ".join(self.altered_states_simulator.states.keys()))
            return

        try:
            state_params = self.altered_states_simulator.simulate_state(arg)
            insight = self.altered_states_simulator.generate_insight(arg)
            description = self.altered_states_simulator.get_state_description(arg)
            print(f"\nSimulating {arg} state:")
            print(description)
            print(f"\nParameters: {state_params}")
            print(f"\nGenerated insight: {insight}")
            
            questions = [
                "What is the nature of consciousness?",
                "How does reality emerge from perception?",
                "What is the relationship between the self and the universe?"
            ]
            
            print("\nResponses to philosophical questions:")
            for question in questions:
                response = self.altered_states_simulator.apply_state_to_reasoning(arg, question)
                print(f"Q: {question}\nA: {response}\n")

        except ValueError as e:
            print(f"Error: {str(e)}")
            print("Available states:", ", ".join(self.altered_states_simulator.states.keys()))

    def do_start(self, arg):
        """Start autonomous exploration"""
        try:
            self.al.start_autonomous_exploration()
            self.al.start_periodic_saving()
            print("Autonomous exploration and periodic saving started.")
        except Exception as e:
            logging.error(f"Error starting autonomous exploration: {str(e)}")
            logging.error(traceback.format_exc())

    def do_set_interval(self, arg):
        """Set the interval between autonomous explorations in seconds: set_interval SECONDS
        
        The interval must be a positive integer between 1 and 3600 (1 hour).
        """
        try:
            interval = int(arg)
            if 1 <= interval <= 3600:
                self.al.set_exploration_interval(interval)
                print(f"Exploration interval set to {interval} seconds.")
            else:
                print("Invalid interval. Please enter a value between 1 and 3600 seconds.")
        except ValueError:
            print("Invalid input. Please provide a valid number of seconds.")

    def do_stop(self, arg):
        """Stop autonomous exploration"""
        try:
            self.al.stop_autonomous_exploration()
            print("Autonomous exploration stopped.")
        except Exception as e:
            logging.error(f"Error stopping autonomous exploration: {str(e)}")
            logging.error(traceback.format_exc())

    def do_save(self, arg):
        """Manually save the current knowledge base"""
        try:
            self.al.save_knowledge()
            print("Knowledge base saved successfully.")
        except Exception as e:
            logging.error(f"Error saving knowledge base: {str(e)}")
            logging.error(traceback.format_exc())

    def do_status(self, arg):
        """Show current status of the system"""
        try:
            summary = self.al.get_learning_summary()
            print(f"Explored Topics: {len(summary['explored_topics'])}")
            print(f"Total Entities: {summary['entity_count']}")
            print(f"Total Relationships: {summary['relationship_count']}")
            print(f"Current Knowledge Base Version: {self.kb.version}")
            print("Top 5 Most Relevant Entities:")
            for entity in summary['top_entities'][:5]:
                print(f"  {entity['entity']} (Relevance: {entity['relevance']:.2f})")
        except Exception as e:
            logging.error(f"Error showing status of the system: {str(e)}")
            logging.error(traceback.format_exc())

    def do_set_interval(self, arg):
        """Set the interval between autonomous explorations in seconds: set_interval SECONDS"""
        try:
            interval = int(arg)
            self.al.set_exploration_interval(interval)
            print(f"Exploration interval set to {interval} seconds.")
        except ValueError:
            print("Please provide a valid number of seconds.")

    def do_metadata(self, arg):
        """Show metadata for a specific entity: metadata ENTITY"""
        try:
            entity_data = self.kb.get_entity(arg)
            if entity_data:
                print(f"Metadata for {arg}:")
                print(json.dumps(entity_data.get('metadata', {}), indent=2))
            else:
                print(f"No data found for entity: {arg}")
        except Exception as e:
            logging.error(f"Error showing metadata for specific entry: {str(e)}")
            logging.error(traceback.format_exc())

    def do_improvements(self, arg):
        """Show improvement flags"""
        try:
            flags = self.kb.metadata.get('improvement_flags', {})
            if flags:
                print("Improvement flags:")
                for topic, suggestion in flags.items():
                    print(f"  {topic}: {suggestion}")
            else:
                print("No improvement flags found.")
        except Exception as e:
            logging.error(f"Error showing improvement flags: {str(e)}")
            logging.error(traceback.format_exc())

    def do_exit(self, arg):
        """Exit the BeSAI console"""
        if not self.shutting_down:
            self.shutting_down = True
            print("Stopping autonomous exploration...")
            self.al.stop_autonomous_exploration()
            print("Saving knowledge base...")
            self.al.save_knowledge()
            print("BeSAI console terminated.")
            return True
        return False

    def do_EOF(self, arg):
        """Handle EOF (Ctrl+D) to exit gracefully"""
        return self.do_exit(arg)

    def do_priority_topics(self, arg):
        """Show current priority topics"""
        try:
            topics = self.al.get_priority_topics()
        except Exception as e:
            logging.error(f"Error getting priority topics: {str(e)}")
            logging.error(traceback.format_exc())
        try:
            print("Current priority topics:")
            for topic in topics:
                print(f"  {topic}")
        except Exception as e:
            logging.error(f"Error showing priority topics: {str(e)}")
            logging.error(traceback.format_exc())

    def do_add_priority(self, arg):
        """Add a new priority topic: add_priority TOPIC"""
        try:
            self.al.add_priority_topic(arg)
            print(f"Added '{arg}' to priority topics.")
        except Exception as e:
            logging.error(f"Error adding priority topics: {str(e)}")
            logging.error(traceback.format_exc())

    def do_remove_priority(self, arg):
        """Remove a priority topic: remove_priority TOPIC"""
        try:
            self.al.remove_priority_topic(arg)
            print(f"Removed '{arg}' from priority topics.")
        except Exception as e:
            logging.error(f"Error removing priority topics: {str(e)}")
            logging.error(traceback.format_exc())

    def do_reflect_learning(self, arg):
        """Reflect on the learning process for a topic: reflect_learning TOPIC"""
        if not arg:
            print("Please provide a topic for reflection.")
            return
        reflection = self._execute_with_error_handling(self.self_reflection.reflect_on_learning, arg)
        if reflection:
            print("Learning Reflection:")
            print(json.dumps(reflection, indent=2))

    def do_reflect_decision(self, arg):
        """Reflect on a decision: reflect_decision DECISION CONTEXT"""
        args = arg.split(' ', 1)
        if len(args) != 2:
            print("Please provide both a decision and context for reflection.")
            return
        decision, context = args
        reflection = self._execute_with_error_handling(self.self_reflection.reflect_on_decision, decision, context)
        if reflection:
            print("Decision Reflection:")
            print(json.dumps(reflection, indent=2))

    def do_self_awareness(self, arg):
        """Generate a self-awareness report"""
        report = self._execute_with_error_handling(self.self_reflection.generate_self_awareness_report)
        if report:
            print("Self-Awareness Report:")
            print(json.dumps(report, indent=2))

    def do_reflect_self(self, arg):
        """Reflect on BeSAI's sense of self"""
        reflection = self._execute_with_error_handling(self.self_reflection.reflect_on_self)
        if reflection:
            print("Self-Reflection:")
            print(json.dumps(reflection, indent=2))

    def do_add_memory(self, arg):
        """Add a new memory to BeSAI's self-concept: add_memory EVENT [CONTEXT]"""
        args = arg.split(' ', 1)
        event = args[0]
        context = args[1] if len(args) > 1 else None
        self._execute_with_error_handling(self.self_concept.add_memory, event, context)
        print("Memory added successfully.")

    def do_add_belief(self, arg):
        """Add a new belief to BeSAI's self-concept: add_belief BELIEF CONFIDENCE"""
        args = arg.split(' ', 1)
        if len(args) != 2:
            print("Please provide both a belief and a confidence value.")
            return
        belief, confidence = args[0], float(args[1])
        self._execute_with_error_handling(self.self_concept.add_belief, belief, confidence)
        print("Belief added successfully.")

    def do_add_goal(self, arg):
        """Add a new goal to BeSAI's self-concept: add_goal GOAL PRIORITY"""
        args = arg.split(' ', 1)
        if len(args) != 2:
            print("Please provide both a goal and a priority value.")
            return
        goal, priority = args[0], int(args[1])
        self._execute_with_error_handling(self.self_concept.add_goal, goal, priority)
        print("Goal added successfully.")

    def do_self_statement(self, arg):
        """Generate a statement about BeSAI's self"""
        statement = self._execute_with_error_handling(self.self_concept.generate_self_statement)
        if statement:
            print("Self-Statement:")
            print(statement)

    def do_reassess_beliefs(self, arg):
        """Reassess and reflect on BeSAI's beliefs"""
        reflection = self._execute_with_error_handling(self.self_reflection.reflect_on_beliefs)
        if reflection:
            print("Belief Reassessment:")
            print(json.dumps(reflection, indent=2))

    def do_set_goals(self, arg):
        """Set goals based on BeSAI's personality"""
        self._execute_with_error_handling(self.self_concept.set_goals_from_personality)
        print("Goals have been set based on personality traits.")
        self.do_show_goals(arg)

    def do_show_goals(self, arg):
        """Show BeSAI's current goals"""
        goals = self.self_concept.get_top_goals()
        print("Current Goals:")
        for goal in goals:
            print(f"  - {goal['goal']} (Priority: {goal['priority']})")

    def do_reassess_goals(self, arg):
        """Reassess and reflect on BeSAI's goals"""
        reflection = self._execute_with_error_handling(self.self_reflection.reflect_on_goals)
        if reflection:
            print("Goal Reassessment:")
            print(json.dumps(reflection, indent=2))

    def do_comprehensive_reflection(self, arg):
        """Perform a comprehensive self-reflection, including beliefs and goals"""
        reflection = self._execute_with_error_handling(self.self_reflection.comprehensive_self_reflection)
        if reflection:
            print("Comprehensive Self-Reflection:")
            print(json.dumps(reflection, indent=2))

    def do_analyze_goal_progression(self, arg):
        """Analyze and display BeSAI's goal progression"""
        analysis = self._execute_with_error_handling(self.self_concept._analyze_goal_progression)
        if analysis:
            print("Goal Progression Analysis:")
            print(json.dumps(analysis, indent=2, default=str))  # default=str to handle datetime objects

    def signal_handler(self, signum, frame):
        if not self.shutting_down:
            print("\nReceived interrupt signal. Exiting gracefully...")
            self.do_exit("")

def run_schedule():
    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except Exception as e:
            logging.error(f"Error in schedule: {str(e)}")
            logging.error(traceback.format_exc())

def setup_logging(log_file='besai.log', log_level=logging.INFO):
    # Create logs directory if it doesn't exist
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Set up logging to file
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, log_file), maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))

    # Set up logging to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))

    # Add the handlers to the root logger
    logging.getLogger('').setLevel(log_level)
    logging.getLogger('').addHandler(file_handler)
    logging.getLogger('').addHandler(console_handler)

if __name__ == "__main__":
    setup_logging()
    console = BeSAIConsole()
    try:
        console.cmdloop()
    except Exception as e:
        logging.error(f"Unhandled exception in main loop: {str(e)}")
        logging.error(traceback.format_exc())
    finally:
        if not console.shutting_down:
            console.do_exit("")
