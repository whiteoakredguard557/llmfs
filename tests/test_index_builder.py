"""
Tests for llmfs.context.index_builder — IndexBuilder.
"""
import pytest
from llmfs import MemoryFS
from llmfs.context.index_builder import (
    IndexBuilder, _extract_turn_id, _artifact_label, _format_timestamp,
)


@pytest.fixture
def mem(tmp_path):
    return MemoryFS(path=tmp_path / "llmfs")


SESSION = "testidx"


class TestEstimateTokens:
    def test_basic(self):
        assert IndexBuilder.estimate_tokens("hello world") == 2  # 11 chars // 4 = 2
        assert IndexBuilder.estimate_tokens("") == 1  # min 1

    def test_longer_text(self):
        text = "a" * 400
        assert IndexBuilder.estimate_tokens(text) == 100


class TestExtractTurnId:
    def test_turns_path(self):
        assert _extract_turn_id("/session/abc/turns/3") == 3

    def test_code_path(self):
        assert _extract_turn_id("/session/abc/code/turn_7_0") == 7

    def test_no_turn(self):
        assert _extract_turn_id("/knowledge/auth/architecture") == -1


class TestArtifactLabel:
    def test_code_label(self):
        class FakeObj:
            tags = ["code", "python", "assistant"]
        assert _artifact_label("/session/s/code/turn_1_0", FakeObj()) == "code:python"

    def test_error_label(self):
        class FakeObj:
            tags = ["error"]
        assert _artifact_label("/session/s/errors/turn_1", FakeObj()) == "error"

    def test_decision_label(self):
        class FakeObj:
            tags = ["decision", "user"]
        assert _artifact_label("/session/s/decisions/turn_1", FakeObj()) == "decision"

    def test_turn_label_user(self):
        class FakeObj:
            tags = ["turn", "user"]
        assert _artifact_label("/session/s/turns/1", FakeObj()) == "user"

    def test_turn_label_assistant(self):
        class FakeObj:
            tags = ["turn", "assistant"]
        assert _artifact_label("/session/s/turns/2", FakeObj()) == "assistant"


class TestFormatTimestamp:
    def test_valid_iso(self):
        result = _format_timestamp("2026-03-18T10:30:00+00:00")
        assert result == "10:30"

    def test_none(self):
        assert _format_timestamp(None) == ""

    def test_empty(self):
        assert _format_timestamp("") == ""

    def test_invalid(self):
        assert _format_timestamp("not-a-date") == ""


class TestIndexBuilder:
    def test_empty_session(self, mem):
        builder = IndexBuilder()
        text = builder.build("emptysess", mem=mem)
        assert "No memories" in text

    def test_header_present(self, mem):
        mem.write(f"/session/{SESSION}/turns/1", "Hello world", layer="session",
                  tags=["turn", "user"])
        builder = IndexBuilder()
        text = builder.build(SESSION, mem=mem)
        assert "## LLMFS Memory Index" in text

    def test_entries_listed(self, mem):
        mem.write(f"/session/{SESSION}/turns/2", "Test content", layer="session",
                  tags=["turn", "assistant"])
        builder = IndexBuilder()
        text = builder.build(SESSION, mem=mem)
        assert f"/session/{SESSION}/turns/2" in text

    def test_truncation_with_omission_note(self, mem):
        # Write 30 memories to trigger truncation at max_entries=10
        for i in range(30):
            mem.write(
                f"/session/{SESSION}/turns/{i}",
                f"Turn {i} content",
                layer="session",
                tags=["turn", "user"],
            )
        builder = IndexBuilder(max_entries=10)
        text = builder.build(SESSION, mem=mem)
        assert "more" in text.lower()

    def test_no_truncation_within_limit(self, mem):
        for i in range(5):
            mem.write(
                f"/session/{SESSION}/turns/{i + 100}",
                f"Turn {i + 100}",
                layer="session",
                tags=["turn", "user"],
            )
        builder = IndexBuilder(max_entries=50)
        text = builder.build(SESSION, mem=mem)
        assert "more" not in text.lower() or "0 more" in text

    def test_max_entries_override(self, mem):
        for i in range(10):
            mem.write(
                f"/session/{SESSION}/turns/{i + 200}",
                f"Turn {i + 200}",
                layer="session",
                tags=["turn", "user"],
            )
        builder = IndexBuilder(max_entries=50)
        # Override max_entries at call time
        text = builder.build(SESSION, mem=mem, max_entries=5)
        assert "more" in text.lower()
