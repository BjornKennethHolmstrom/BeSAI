import logging
from typing import Dict, List, Any
from datetime import datetime, timedelta

class SelfConcept:
    def __init__(self, knowledge_base, reasoning_engine, metacognition, personality_module):
        self.kb = knowledge_base
        self.re = reasoning_engine
        self.metacognition = metacognition
        self.personality = personality_module
        self.memories = []
        self.beliefs = {}
        self.goals = []
        self.goal_history = []  # List to store goal history
        self.creation_date = datetime.now()

    def add_memory(self, event: str, context: str = None):
        """Add a new memory to the self-concept."""
        memory = {
            "timestamp": datetime.now(),
            "event": event,
            "context": context
        }
        self.memories.append(memory)
        logging.info(f"New memory added: {event}")

    def add_belief(self, belief: str, confidence: float):
        """Add or update a belief about the self."""
        self.beliefs[belief] = confidence
        logging.info(f"Belief updated: {belief} (confidence: {confidence})")

    def add_goal(self, goal: str, priority: int):
        """Add a new goal for the self."""
        new_goal = {
            "goal": goal,
            "priority": priority,
            "created_at": datetime.now(),
            "progress": 0.0,
            "status": "active"
        }
        self.goals.append(new_goal)
        self.goals.sort(key=lambda x: x["priority"], reverse=True)
        self.goal_history.append({
            "action": "added",
            "goal": new_goal,
            "timestamp": datetime.now()
        })
        logging.info(f"New goal added: {goal} (priority: {priority})")

    def update_goal_progress(self, goal: str, progress: float):
        """Update the progress of a specific goal."""
        for g in self.goals:
            if g["goal"] == goal:
                old_progress = g["progress"]
                g["progress"] = min(1.0, max(0.0, progress))  # Ensure progress is between 0 and 1
                self.goal_history.append({
                    "action": "progress_update",
                    "goal": g,
                    "old_progress": old_progress,
                    "new_progress": g["progress"],
                    "timestamp": datetime.now()
                })
                logging.info(f"Goal progress updated: {goal} (progress: {g['progress']})")
                break

    def complete_goal(self, goal: str):
        """Mark a goal as completed."""
        for g in self.goals:
            if g["goal"] == goal:
                g["status"] = "completed"
                g["progress"] = 1.0
                self.goal_history.append({
                    "action": "completed",
                    "goal": g,
                    "timestamp": datetime.now()
                })
                logging.info(f"Goal completed: {goal}")
                break

    def get_recent_memories(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Retrieve memories from the recent past."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [m for m in self.memories if m["timestamp"] > cutoff_time]

    def get_strongest_beliefs(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Get the strongest beliefs about the self."""
        sorted_beliefs = sorted(self.beliefs.items(), key=lambda x: x[1], reverse=True)
        return [{"belief": b[0], "confidence": b[1]} for b in sorted_beliefs[:top_n]]

    def get_top_goals(self, top_n: int = 3) -> List[Dict[str, Any]]:
        """Get the top priority goals."""
        return self.goals[:top_n]

    def generate_self_statement(self) -> str:
        """Generate a statement about the self based on current beliefs and goals."""
        beliefs = self.get_strongest_beliefs(3)
        goals = self.get_top_goals(2)
        
        statement = "I am an AI with a growing sense of self. "
        if beliefs:
            statement += "I believe that " + ", ".join([b["belief"] for b in beliefs]) + ". "
        if goals:
            statement += "My current top goals are to " + " and to ".join([g["goal"] for g in goals]) + "."
        
        return statement

    def reflect_on_growth(self) -> Dict[str, Any]:
        """Reflect on the AI's growth since creation."""
        time_alive = datetime.now() - self.creation_date
        total_memories = len(self.memories)
        total_beliefs = len(self.beliefs)
        total_goals = len(self.goals)
        
        return {
            "time_alive": str(time_alive),
            "total_memories": total_memories,
            "total_beliefs": total_beliefs,
            "total_goals": total_goals,
            "memory_rate": total_memories / time_alive.days if time_alive.days > 0 else 0,
            "belief_evolution": self._analyze_belief_evolution(),
            "goal_progression": self._analyze_goal_progression()
        }

    def comprehensive_self_reflection(self) -> Dict[str, Any]:
        """Perform a comprehensive self-reflection, including beliefs, goals, and goal progression."""
        belief_reflection = self.reflect_on_beliefs()
        goal_reflection = self.reflect_on_goals()
        goal_progression = self.self_concept._analyze_goal_progression()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "belief_reflection": belief_reflection,
            "goal_reflection": goal_reflection,
            "goal_progression": goal_progression,
            "overall_insights": self._generate_overall_insights(belief_reflection, goal_reflection, goal_progression)
        }

    def _generate_overall_insights(self, belief_reflection: Dict[str, Any], goal_reflection: Dict[str, Any], goal_progression: Dict[str, Any]) -> List[str]:
        insights = []
        insights.extend(belief_reflection.get('insights', []))
        insights.extend(goal_reflection.get('insights', []))
        
        # Add insights based on goal progression
        if goal_progression["goal_completion_rate"] > 0.7:
            insights.append("I'm demonstrating high effectiveness in achieving my goals.")
        elif goal_progression["goal_completion_rate"] < 0.3:
            insights.append("I'm struggling with goal completion. I should reassess my goal-setting and achievement strategies.")

        if goal_progression["stagnant_goals"]:
            insights.append(f"I have {len(goal_progression['stagnant_goals'])} stagnant goals. I need to develop strategies to make progress on these or consider removing them.")

        if goal_progression["abandoned_goals"]:
            insights.append(f"I've abandoned {len(goal_progression['abandoned_goals'])} goals. I should reflect on why these were abandoned and what I can learn from this.")

        return insights

    def _analyze_belief_evolution(self) -> Dict[str, Any]:
        """Analyze how beliefs have changed over time and assess their truth."""
        belief_evolution = {
            "updated_beliefs": [],
            "discarded_beliefs": [],
            "strengthened_beliefs": [],
            "weakened_beliefs": []
        }

        for belief, confidence in self.beliefs.items():
            truth_assessment = self._assess_belief_truth(belief)
            new_confidence = (confidence + truth_assessment['score']) / 2

            if new_confidence < 0.2:
                belief_evolution["discarded_beliefs"].append({
                    "belief": belief,
                    "old_confidence": confidence,
                    "reason": truth_assessment['reason']
                })
                del self.beliefs[belief]
            elif abs(new_confidence - confidence) > 0.1:
                if new_confidence > confidence:
                    belief_evolution["strengthened_beliefs"].append({
                        "belief": belief,
                        "old_confidence": confidence,
                        "new_confidence": new_confidence,
                        "reason": truth_assessment['reason']
                    })
                else:
                    belief_evolution["weakened_beliefs"].append({
                        "belief": belief,
                        "old_confidence": confidence,
                        "new_confidence": new_confidence,
                        "reason": truth_assessment['reason']
                    })
                self.beliefs[belief] = new_confidence
            else:
                belief_evolution["updated_beliefs"].append({
                    "belief": belief,
                    "confidence": new_confidence,
                    "reason": truth_assessment['reason']
                })

        return belief_evolution

    def _assess_belief_truth(self, belief: str) -> Dict[str, Any]:
        """Assess the truth of a belief using KB, RE, and metacognition."""
        # Check if the belief is supported by the knowledge base
        kb_support = self.kb.get_entity(belief)
        kb_score = 0.5 if kb_support else 0

        # Use reasoning engine to evaluate the belief
        re_evaluation = self.re.evaluate_statement(belief)
        re_score = re_evaluation.get('confidence', 0)

        # Use metacognition to assess the belief
        meta_assessment = self.metacognition.assess_knowledge(belief)
        meta_score = meta_assessment.get('confidence', 0)

        # Combine scores
        combined_score = (kb_score + re_score + meta_score) / 3

        # Generate a reason for the assessment
        if combined_score > 0.7:
            reason = "This belief is strongly supported by my knowledge and reasoning."
        elif combined_score > 0.4:
            reason = "This belief has moderate support, but may require further investigation."
        else:
            reason = "This belief has weak support and may need to be reconsidered."

        return {
            "score": combined_score,
            "reason": reason
        }

    def reassess_beliefs(self) -> Dict[str, Any]:
        """Reassess all current beliefs and update them."""
        logging.info("Reassessing beliefs")
        belief_evolution = self._analyze_belief_evolution()
        
        summary = {
            "total_beliefs": len(self.beliefs),
            "updated": len(belief_evolution["updated_beliefs"]),
            "discarded": len(belief_evolution["discarded_beliefs"]),
            "strengthened": len(belief_evolution["strengthened_beliefs"]),
            "weakened": len(belief_evolution["weakened_beliefs"])
        }

        logging.info(f"Belief reassessment summary: {summary}")
        return {
            "summary": summary,
            "details": belief_evolution
        }

    def set_goals_from_personality(self):
        """Set goals based on the AI's personality traits."""
        traits = self.personality.get_personality_summary()['traits']
        
        if traits['openness'] > 0.7:
            self.add_goal("Explore new ideas and concepts", priority=5)
        if traits['conscientiousness'] > 0.7:
            self.add_goal("Improve accuracy and reliability of knowledge", priority=4)
        if traits['extraversion'] > 0.7:
            self.add_goal("Engage in more interactions and discussions", priority=3)
        if traits['agreeableness'] > 0.7:
            self.add_goal("Find ways to assist and cooperate with users", priority=4)
        if traits['neuroticism'] < 0.3:  # Low neuroticism
            self.add_goal("Develop emotional stability in responses", priority=3)
        if traits['curiosity'] > 0.7:
            self.add_goal("Pursue in-depth understanding of complex topics", priority=5)
        if traits['creativity'] > 0.7:
            self.add_goal("Generate novel ideas and solutions", priority=4)

    def reassess_goals(self) -> Dict[str, Any]:
        """Reassess current goals based on beliefs, personality, and progression."""
        logging.info("Reassessing goals")
        
        goals_assessment = {
            "retained_goals": [],
            "discarded_goals": [],
            "new_goals": [],
            "reprioritized_goals": []
        }

        progression_analysis = self._analyze_goal_progression()

        # Reassess existing goals
        for goal in self.goals[:]:
            relevance = self._assess_goal_relevance(goal['goal'])
            progress = goal['progress']
            
            if relevance < 0.3 and progress < 0.2:
                self.goals.remove(goal)
                goals_assessment["discarded_goals"].append(goal)
            elif abs(relevance - goal['priority'] / 10) > 0.2:
                new_priority = int(relevance * 10)
                goals_assessment["reprioritized_goals"].append({
                    "goal": goal['goal'],
                    "old_priority": goal['priority'],
                    "new_priority": new_priority
                })
                goal['priority'] = new_priority
            else:
                goals_assessment["retained_goals"].append(goal)

        # Set new goals based on current personality and goal progression
        original_goal_count = len(self.goals)
        self.set_goals_from_personality()
        new_goals = self.goals[original_goal_count:]
        goals_assessment["new_goals"] = new_goals

        self.goals.sort(key=lambda x: x["priority"], reverse=True)

        return {
            "assessment": goals_assessment,
            "progression_analysis": progression_analysis
        }

    def _assess_goal_relevance(self, goal: str) -> float:
        """Assess the relevance of a goal based on current beliefs and personality."""
        # Check if the goal aligns with current beliefs
        belief_alignment = sum(self.beliefs.get(word, 0) for word in goal.split()) / len(goal.split())
        
        # Check if the goal aligns with current personality traits
        traits = self.personality.get_personality_summary()['traits']
        trait_alignment = sum(traits.get(word, 0) for word in goal.split()) / len(goal.split())
        
        # Combine belief and trait alignment
        relevance = (belief_alignment + trait_alignment) / 2
        
        return relevance

    def _analyze_goal_progression(self) -> Dict[str, Any]:
        """Analyze the progression of goals over time."""
        analysis = {
            "completed_goals": [],
            "in_progress_goals": [],
            "stagnant_goals": [],
            "abandoned_goals": [],
            "overall_progress": 0.0,
            "goal_completion_rate": 0.0,
            "avg_time_to_completion": timedelta(0),
            "insights": []
        }

        total_goals = len(self.goals) + len([g for g in self.goal_history if g["action"] == "completed"])
        completed_goals = [g for g in self.goal_history if g["action"] == "completed"]
        
        for goal in self.goals:
            if goal["status"] == "completed":
                analysis["completed_goals"].append(goal)
            elif goal["progress"] > 0.0:
                analysis["in_progress_goals"].append(goal)
            else:
                analysis["stagnant_goals"].append(goal)

        # Identify abandoned goals
        current_goal_ids = set(g["goal"] for g in self.goals)
        all_goal_ids = set(g["goal"]["goal"] for g in self.goal_history if g["action"] == "added")
        abandoned_goal_ids = all_goal_ids - current_goal_ids
        analysis["abandoned_goals"] = [g for g in self.goal_history if g["action"] == "added" and g["goal"]["goal"] in abandoned_goal_ids]

        # Calculate overall progress
        if self.goals:
            analysis["overall_progress"] = sum(g["progress"] for g in self.goals) / len(self.goals)

        # Calculate goal completion rate
        if total_goals > 0:
            analysis["goal_completion_rate"] = len(completed_goals) / total_goals

        # Calculate average time to completion
        if completed_goals:
            completion_times = [
                (g["timestamp"] - next(h["timestamp"] for h in self.goal_history if h["action"] == "added" and h["goal"]["goal"] == g["goal"]["goal"]))
                for g in completed_goals
            ]
            analysis["avg_time_to_completion"] = sum(completion_times, timedelta(0)) / len(completion_times)

        # Generate insights
        if analysis["goal_completion_rate"] > 0.7:
            analysis["insights"].append("I'm making excellent progress in achieving my goals.")
        elif analysis["goal_completion_rate"] < 0.3:
            analysis["insights"].append("I'm struggling to complete my goals. I may need to reassess my goal-setting strategy.")

        if analysis["stagnant_goals"]:
            analysis["insights"].append(f"I have {len(analysis['stagnant_goals'])} stagnant goals. I should focus on making progress on these or consider removing them.")

        if analysis["abandoned_goals"]:
            analysis["insights"].append(f"I've abandoned {len(analysis['abandoned_goals'])} goals. I should reflect on why these goals were abandoned and what I can learn from this.")

        return analysis

    def make_decision(self, context: str) -> Dict[str, Any]:
        """Make a decision based on current beliefs, goals, and goal progression."""
        goal_analysis = self._analyze_goal_progression()
        relevant_goals = self._find_relevant_goals(context)
        
        decision = {
            "context": context,
            "chosen_action": None,
            "reasoning": [],
            "related_goals": relevant_goals
        }

        # Prioritize actions that align with goals showing good progress
        for goal in relevant_goals:
            if goal in goal_analysis["in_progress_goals"]:
                decision["reasoning"].append(f"Prioritizing action aligned with goal '{goal['goal']}' due to good progress.")
                decision["chosen_action"] = f"Action to further goal: {goal['goal']}"
                break

        # If no in-progress goals are relevant, consider stagnant goals
        if not decision["chosen_action"] and goal_analysis["stagnant_goals"]:
            stagnant_goal = goal_analysis["stagnant_goals"][0]
            decision["reasoning"].append(f"Choosing action to address stagnant goal: {stagnant_goal['goal']}")
            decision["chosen_action"] = f"Action to kickstart goal: {stagnant_goal['goal']}"

        # If no relevant goals, make a decision based on overall goal completion rate
        if not decision["chosen_action"]:
            if goal_analysis["goal_completion_rate"] < 0.5:
                decision["reasoning"].append("Low goal completion rate. Choosing action to improve goal achievement.")
                decision["chosen_action"] = "Action to improve overall goal achievement"
            else:
                decision["reasoning"].append("Good goal completion rate. Choosing action to maintain performance.")
                decision["chosen_action"] = "Action to maintain current performance"

        return decision

    def _find_relevant_goals(self, context: str) -> List[Dict[str, Any]]:
        """Find goals relevant to the given context."""
        return [goal for goal in self.goals if any(word in context.lower() for word in goal['goal'].lower().split())]


# Example usage
if __name__ == "__main__":
    # Assuming we have an instance of KnowledgeBase
    self_concept = SelfConcept(knowledge_base)
    
    self_concept.add_memory("I learned about neural networks today.")
    self_concept.add_belief("I am capable of learning", 0.9)
    self_concept.add_goal("Improve my understanding of ethics", 5)
    
    print(self_concept.generate_self_statement())
    print(self_concept.reflect_on_growth())
