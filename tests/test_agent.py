import unittest
from agent.core import DecideItAgent, AgentConfig


class FakeLLM:
    def generate(self, messages, json_mode=False):
        return '{"summary":"mock result"}'


class AgentTests(unittest.TestCase):
    def test_plan_supports_missing_solution(self):
        result = DecideItAgent().evaluate({"project_title": "River",
            "problem_description": "Plastic is entering the river."})
        self.assertTrue(any(x["action"] == "generate_solutions" for x in result.plan))

    def test_llm_json_synthesis(self):
        result = DecideItAgent(FakeLLM()).evaluate({"project_title": "River",
            "problem_description": "Plastic is entering the river.",
            "proposed_solution": "Collection points"})
        self.assertEqual(result.output["summary"], "mock result")
        self.assertEqual(result.status, "completed")

    def test_iteration_limit(self):
        agent = DecideItAgent(config=AgentConfig(max_iterations=2))
        result = agent.evaluate({"project_title": "River",
            "problem_description": "Plastic is entering the river."})
        self.assertEqual(len(result.history), 4)  # planned + not_configured per step

    def test_rejects_empty_problem(self):
        with self.assertRaises(ValueError):
            DecideItAgent().evaluate({"problem_description": " "})


if __name__ == "__main__":
    unittest.main()
