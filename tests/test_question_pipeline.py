import unittest
from unittest.mock import patch

from noelle import Arch, Choice, NoellePipeline, Noul, Score
from noelle.pipeline import Decision


class DecisionStub:
    arch = Arch.SmollmDB135m

    def __init__(self):
        self.calls = []

    def predict(self, *, state, question, options, kind):
        self.calls.append((state, question, options, kind))
        probabilities = {
            "choice": (0.6, 0.3, 0.1),
            "noul": (0.8, 0.2),
            "score": (0.1, 0.2, 0.7),
        }[kind]
        return Decision(tuple(options), probabilities, max(range(len(probabilities)), key=probabilities.__getitem__), False, 0)


class QuestionPipelineTest(unittest.TestCase):
    def setUp(self):
        self.stub = DecisionStub()
        with patch("noelle.pipeline.load_pipeline", return_value=self.stub):
            self.pipeline = NoellePipeline(Arch.SmollmDB135m)

    def test_named_questions_keep_ids_out_of_prompts_and_return_typed_answers(self):
        state = {"document": "I was charged twice."}
        response = self.pipeline(
            state=state,
            questions={
                "category": Choice(
                    instructions="What is this about?",
                    criteria={"billing": None, "technical": "A broken feature", "other": None},
                ),
                "refund": Noul(
                    instructions="Does the customer request a refund?",
                    criteria={"false": "No refund request", "true": "Requests money back"},
                ),
                "urgency": Score(
                    instructions="How urgent?", criteria=["can wait", "this week", "today"],
                ),
            },
        )
        self.assertEqual(response.choices["category"].choice, "billing")
        self.assertEqual(response.choices["category"].probabilities,
                         {"billing": 0.6, "technical": 0.3, "other": 0.1})
        self.assertAlmostEqual(response.nouls["refund"].noul, 0.8)
        self.assertAlmostEqual(response.scores["urgency"].score, 1.6)
        self.assertEqual(response.scores["urgency"].probabilities,
                         {0: 0.1, 1: 0.2, 2: 0.7})
        self.assertEqual(response.scores["urgency"].legend,
                         {0: "can wait", 1: "this week", 2: "today"})
        self.assertEqual(self.stub.calls, [
            (state, "What is this about?", ["billing", "technical: A broken feature", "other"], "choice"),
            (state, "Does the customer request a refund?",
             ["agree with the instruction: Requests money back",
              "disagree with the instruction: No refund request"], "noul"),
            (state, "How urgent?", ["can wait", "this week", "today"], "score"),
        ])

    def test_invalid_question_declarations_fail_before_inference(self):
        with self.assertRaisesRegex(ValueError, "nonempty mapping"):
            self.pipeline(state="x", questions={})
        with self.assertRaisesRegex(ValueError, "at least two"):
            self.pipeline(state="x", questions={"category": Choice("Choose", {"only": None})})
        with self.assertRaisesRegex(ValueError, "false.*true"):
            self.pipeline(state="x", questions={"flag": Noul("Flag?", {"maybe": "?"})})
        with self.assertRaisesRegex(ValueError, "at least two"):
            self.pipeline(state="x", questions={"rating": Score("Rate?", ["one"])})
        self.assertEqual(self.stub.calls, [])


if __name__ == "__main__":
    unittest.main()
