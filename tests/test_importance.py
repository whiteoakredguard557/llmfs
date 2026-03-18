"""
Tests for llmfs.context.importance — ImportanceScorer.
"""
import pytest
from llmfs.context.importance import ImportanceScorer, ImportanceWeights, _is_filler, _token_count


class TestImportanceWeights:
    def test_defaults_valid(self):
        w = ImportanceWeights()
        assert 0 < w.base <= 1
        assert w.code_block_boost > 0
        assert w.error_boost > 0

    def test_custom_weights(self):
        w = ImportanceWeights(base=0.3, code_block_boost=0.1)
        assert w.base == 0.3
        assert w.code_block_boost == 0.1


class TestImportanceScorer:
    def setup_method(self):
        self.scorer = ImportanceScorer()

    def test_base_score(self):
        # No boosts, no penalties
        score = self.scorer.score("A neutral, moderate-length sentence here.",
                                  role="assistant", turn_index=0, total_turns=10)
        assert 0.0 <= score <= 1.0

    def test_code_block_boost(self):
        plain = self.scorer.score("Some text.", role="assistant", turn_index=5, total_turns=10)
        with_code = self.scorer.score("```python\nprint('hi')\n```",
                                      role="assistant", turn_index=5, total_turns=10)
        assert with_code > plain

    def test_error_boost(self):
        plain = self.scorer.score("Everything is fine.", role="assistant",
                                  turn_index=5, total_turns=10)
        with_err = self.scorer.score(
            "Traceback (most recent call last):\n  File 'a.py', line 1\nTypeError: oops",
            role="assistant", turn_index=5, total_turns=10,
        )
        assert with_err > plain

    def test_decision_keyword_boost(self):
        plain = self.scorer.score("Let me think about this.", role="assistant",
                                  turn_index=5, total_turns=10)
        with_decision = self.scorer.score("I decided to use PostgreSQL for the database.",
                                          role="assistant", turn_index=5, total_turns=10)
        assert with_decision > plain

    def test_user_role_boost(self):
        assistant_score = self.scorer.score("Hello there.", role="assistant",
                                            turn_index=5, total_turns=10)
        user_score = self.scorer.score("Hello there.", role="user",
                                       turn_index=5, total_turns=10)
        assert user_score > assistant_score

    def test_recency_boost(self):
        # Last turn gets recency boost
        early = self.scorer.score("Some content.", role="assistant",
                                  turn_index=0, total_turns=10)
        recent = self.scorer.score("Some content.", role="assistant",
                                   turn_index=9, total_turns=10)
        assert recent > early

    def test_filler_penalty(self):
        filler = self.scorer.score("ok", role="user", turn_index=5, total_turns=10)
        non_filler = self.scorer.score("Please fix the authentication module.",
                                       role="user", turn_index=5, total_turns=10)
        assert filler < non_filler

    def test_short_content_penalty(self):
        short = self.scorer.score("Hi", role="assistant", turn_index=5, total_turns=10)
        long = self.scorer.score(" ".join(["word"] * 100), role="assistant",
                                 turn_index=5, total_turns=10)
        # short should have lower or equal score vs long
        assert short <= long + 0.01  # allow floating tolerance

    def test_score_clamped_to_unit_interval(self):
        # Maximum boosts should not exceed 1.0
        score = self.scorer.score(
            "```python\ncode\n``` Traceback: Error decided will use",
            role="user", turn_index=9, total_turns=10,
        )
        assert 0.0 <= score <= 1.0

    def test_score_batch(self):
        turns = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "```python\nprint('hi')\n```"},
            {"role": "user", "content": "ok"},
        ]
        scores = self.scorer.score_batch(turns)
        assert len(scores) == 3
        # Code turn should score higher than filler
        assert scores[1] > scores[2]

    def test_score_batch_empty(self):
        assert self.scorer.score_batch([]) == []

    def test_multiple_boosts_additive(self):
        """Code + error + decision + user + recent should stack."""
        w = ImportanceWeights(
            base=0.5,
            code_block_boost=0.10,
            error_boost=0.10,
            decision_boost=0.10,
        )
        scorer = ImportanceScorer(weights=w)
        score = scorer.score(
            "```\ncode\n``` Traceback: decided to fix it",
            role="user",
            turn_index=9,
            total_turns=10,
        )
        assert score >= 0.80  # 0.5 + 0.10 + 0.10 + 0.10 + user + recency


class TestHelpers:
    def test_token_count(self):
        assert _token_count("one two three") == 3
        assert _token_count("") == 0

    def test_is_filler_true(self):
        assert _is_filler("ok")
        assert _is_filler("sure")
        assert _is_filler("thanks")
        assert _is_filler("ok.")

    def test_is_filler_false(self):
        assert not _is_filler("please fix the bug")
        assert not _is_filler("the function is broken at line 45")
