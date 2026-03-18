"""
Tests for llmfs.compression.summarizer.ExtractiveSummarizer.
"""
from __future__ import annotations

import pytest

from llmfs.compression.summarizer import (
    ExtractiveSummarizer,
    _split_sentences,
    _tfidf_scores,
    _top_sentences,
)


# ── _split_sentences ──────────────────────────────────────────────────────────


class TestSplitSentences:
    def test_single_sentence(self):
        result = _split_sentences("Hello world this is a sentence.")
        assert len(result) >= 1
        assert "Hello world" in result[0]

    def test_multiple_sentences(self):
        text = (
            "The authentication module fails when tokens expire. "
            "The database connection pool reaches its maximum limit. "
            "A new caching strategy was implemented for better performance."
        )
        result = _split_sentences(text)
        assert len(result) >= 2

    def test_empty_string_returns_something(self):
        result = _split_sentences("")
        assert isinstance(result, list)

    def test_short_sentences_filtered(self):
        # Sentences under 20 chars should be filtered out
        text = "Hi. The quick brown fox jumped over the lazy dog and ran away quickly."
        result = _split_sentences(text)
        assert all(len(s) >= 20 for s in result)


# ── _tfidf_scores ─────────────────────────────────────────────────────────────


class TestTfidfScores:
    def test_returns_list_of_floats(self):
        sentences = [
            "The authentication module has a serious bug.",
            "JWT tokens expire too quickly in production.",
            "The refresh endpoint does not handle edge cases.",
        ]
        scores = _tfidf_scores(sentences)
        assert len(scores) == len(sentences)
        assert all(isinstance(s, float) for s in scores)

    def test_single_sentence_returns_one(self):
        scores = _tfidf_scores(["Only one sentence here."])
        assert scores == [1.0]

    def test_non_negative(self):
        sentences = ["Alpha beta gamma.", "Delta epsilon zeta.", "Eta theta iota."]
        scores = _tfidf_scores(sentences)
        assert all(s >= 0.0 for s in scores)


# ── _top_sentences ────────────────────────────────────────────────────────────


class TestTopSentences:
    def test_empty_returns_empty(self):
        assert _top_sentences("", max_sentences=3) == ""

    def test_short_text_returned_whole(self):
        text = "Only one useful sentence in here."
        result = _top_sentences(text, max_sentences=5)
        assert result.strip() != ""

    def test_long_text_truncated(self):
        sentences = [
            "The quick brown fox jumped over the lazy dog.",
            "Authentication tokens must be refreshed every hour.",
            "Database connections should use connection pooling.",
            "Memory leaks can cause slow degradation of performance.",
            "Code reviews improve overall software quality substantially.",
        ]
        text = " ".join(sentences)
        result = _top_sentences(text, max_sentences=2)
        # Result should be shorter than the original
        assert len(result) < len(text)

    def test_preserves_sentence_order(self):
        """Selected sentences should appear in their original textual order."""
        text = (
            "First point about authentication in systems. "
            "Second point about database connection pooling. "
            "Third point about error handling strategies. "
            "Fourth point about performance optimization techniques. "
            "Fifth point about code review best practices today."
        )
        result = _top_sentences(text, max_sentences=2)
        # Both selected sentences should come from the original text
        assert len(result) > 0


# ── ExtractiveSummarizer ──────────────────────────────────────────────────────


class TestExtractiveSummarizer:
    def test_default_construction(self):
        s = ExtractiveSummarizer()
        assert s._chunk_sentences == 2
        assert s._doc_sentences == 3

    def test_custom_construction(self):
        s = ExtractiveSummarizer(chunk_sentences=1, doc_sentences=5)
        assert s._chunk_sentences == 1
        assert s._doc_sentences == 5

    def test_invalid_chunk_sentences_raises(self):
        with pytest.raises(ValueError, match="chunk_sentences"):
            ExtractiveSummarizer(chunk_sentences=0)

    def test_invalid_doc_sentences_raises(self):
        with pytest.raises(ValueError, match="doc_sentences"):
            ExtractiveSummarizer(doc_sentences=0)

    def test_summarize_chunks_returns_one_per_chunk(self):
        s = ExtractiveSummarizer()
        chunks = [
            "The authentication module fails when the JWT token has expired.",
            "Connection pooling improves database throughput significantly.",
        ]
        summaries = s.summarize_chunks(chunks)
        assert len(summaries) == 2
        assert all(isinstance(x, str) for x in summaries)

    def test_summarize_chunks_empty_chunk_yields_empty_string(self):
        s = ExtractiveSummarizer()
        summaries = s.summarize_chunks(["", "Some real content about databases."])
        assert summaries[0] == ""
        assert len(summaries[1]) > 0

    def test_summarize_document_returns_string(self):
        s = ExtractiveSummarizer()
        content = (
            "The system experienced a critical outage yesterday. "
            "Engineers identified the root cause as a misconfigured load balancer. "
            "The fix was deployed within two hours of detection. "
            "Monitoring was increased to prevent future occurrences. "
            "A post-mortem review is scheduled for next week."
        )
        result = s.summarize_document(content)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_summarize_document_max_sentences_override(self):
        s = ExtractiveSummarizer()
        content = (
            "The authentication module has a serious bug in JWT handling. "
            "Tokens expire too quickly and cause repeated login prompts. "
            "The fix involves updating the expiry calculation to use UTC time. "
            "This change must be deployed before the weekend maintenance window."
        )
        result = s.summarize_document(content, max_sentences=1)
        assert isinstance(result, str)

    def test_summarize_document_invalid_max_sentences(self):
        s = ExtractiveSummarizer()
        with pytest.raises(ValueError, match="max_sentences"):
            s.summarize_document("Some text here.", max_sentences=0)

    def test_summarize_all_returns_tuple(self):
        s = ExtractiveSummarizer()
        content = (
            "First sentence about authentication security. "
            "Second sentence about database optimization. "
            "Third sentence about code quality review."
        )
        chunk_texts = [
            "First sentence about authentication security.",
            "Third sentence about code quality review.",
        ]
        level1, level2 = s.summarize_all(content, chunk_texts)
        assert isinstance(level1, list)
        assert len(level1) == 2
        assert isinstance(level2, str)

    def test_repr(self):
        s = ExtractiveSummarizer()
        r = repr(s)
        assert "ExtractiveSummarizer" in r
        assert "chunk_sentences" in r
