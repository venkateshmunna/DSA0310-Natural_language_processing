import unittest
from unittest import mock

import requests

import answer_generator


class AnswerGeneratorTests(unittest.TestCase):
    def make_result(self, filename="slide.png", caption="A slide", ocr="Quarterly revenue is 25 percent"):
        return {
            "source": filename,
            "filename": filename,
            "timestamp_seconds": 65,
            "caption": caption,
            "ocr_text": ocr,
        }

    def test_list_models_returns_downloaded_ollama_models(self):
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "models": [{"name": "llama3.2:3b"}, {"name": "gemma2:2b"}]
        }
        with mock.patch.object(answer_generator.requests, "get", return_value=response):
            models = answer_generator.list_models()

        self.assertEqual(models, ["llama3.2:3b", "gemma2:2b"])

    def test_list_models_reports_offline_ollama(self):
        with mock.patch.object(
            answer_generator.requests,
            "get",
            side_effect=requests.ConnectionError("offline"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Cannot reach Ollama"):
                answer_generator.list_models()

    def test_insufficient_evidence_does_not_call_ollama(self):
        weak_results = [self.make_result(caption="a", ocr="")]
        with mock.patch.object(answer_generator.requests, "post") as post:
            answer, sources = answer_generator.generate_answer(
                "What is the revenue?",
                weak_results,
            )

        post.assert_not_called()
        self.assertIn("Insufficient evidence", answer)
        self.assertEqual(sources, [])

    def test_generate_answer_posts_grounding_prompt_and_extracts_citations(self):
        results = [
            self.make_result(),
            self.make_result(filename="lecture.mp4", caption="A presenter", ocr="Training begins at 2 PM"),
        ]
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "message": {
                "content": "Revenue increased by 25 percent [1]."
            }
        }

        with mock.patch.object(answer_generator.requests, "post", return_value=response) as post:
            answer, cited_sources = answer_generator.generate_answer(
                "What is the revenue increase?",
                results,
                model="qwen2.5:3b",
            )

        self.assertIn("Revenue increased", answer)
        self.assertEqual(cited_sources, [results[0]])
        call_payload = post.call_args.kwargs["json"]
        self.assertEqual(call_payload["model"], "qwen2.5:3b")
        prompt = call_payload["messages"][1]["content"]
        self.assertIn("Do not use outside knowledge", prompt)
        self.assertIn("Insufficient evidence", prompt)
        self.assertIn("timestamp", prompt.lower())
        self.assertIn("Quarterly revenue", prompt)

    def test_ollama_timeout_has_actionable_error(self):
        results = [self.make_result()]
        with mock.patch.object(
            answer_generator.requests,
            "post",
            side_effect=requests.Timeout("slow"),
        ):
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                answer_generator.generate_answer(
                    "What revenue is shown?",
                    results,
                )

    def test_empty_ollama_answer_is_rejected(self):
        results = [self.make_result()]
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"message": {"content": "  "}}
        with mock.patch.object(answer_generator.requests, "post", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "empty answer"):
                answer_generator.generate_answer(
                    "What is visible?",
                    results,
                )


if __name__ == "__main__":
    unittest.main()
