"""Tests for llmfs.core.memory_layers"""
from datetime import datetime, timedelta, timezone

import pytest
from llmfs.core.memory_layers import (
    LAYER_DEFAULT_TTL,
    VALID_LAYERS,
    MemoryLayer,
    is_expired,
    ttl_expires_at,
)
from llmfs.core.memory_object import MemoryMetadata, MemoryObject


class TestMemoryLayer:
    def test_values(self):
        assert MemoryLayer.SHORT_TERM == "short_term"
        assert MemoryLayer.SESSION == "session"
        assert MemoryLayer.KNOWLEDGE == "knowledge"
        assert MemoryLayer.EVENTS == "events"

    def test_valid_layers_set(self):
        assert "short_term" in VALID_LAYERS
        assert "knowledge" in VALID_LAYERS
        assert "invalid" not in VALID_LAYERS

    def test_default_ttl_short_term(self):
        assert LAYER_DEFAULT_TTL[MemoryLayer.SHORT_TERM] == 60

    def test_default_ttl_permanent(self):
        assert LAYER_DEFAULT_TTL[MemoryLayer.KNOWLEDGE] is None
        assert LAYER_DEFAULT_TTL[MemoryLayer.EVENTS] is None


class TestTTLExpiresAt:
    def test_no_expiry_for_knowledge(self):
        assert ttl_expires_at("knowledge") is None

    def test_expiry_for_short_term(self):
        result = ttl_expires_at("short_term")
        assert result is not None
        expiry = datetime.fromisoformat(result)
        diff = expiry - datetime.now(timezone.utc)
        assert 55 < diff.total_seconds() / 60 < 65

    def test_explicit_override(self):
        result = ttl_expires_at("knowledge", ttl_minutes=10)
        assert result is not None
        expiry = datetime.fromisoformat(result)
        diff = expiry - datetime.now(timezone.utc)
        assert 8 < diff.total_seconds() / 60 < 12

    def test_zero_means_no_expiry(self):
        assert ttl_expires_at("short_term", ttl_minutes=0) is None


class TestIsExpired:
    def _obj(self, ttl_str) -> MemoryObject:
        return MemoryObject(
            id="x", path="/x", content="y", layer="knowledge",
            metadata=MemoryMetadata(ttl=ttl_str),
        )

    def test_no_ttl_not_expired(self):
        assert not is_expired(self._obj(None))

    def test_future_ttl_not_expired(self):
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        assert not is_expired(self._obj(future))

    def test_past_ttl_is_expired(self):
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        assert is_expired(self._obj(past))

    def test_invalid_ttl_not_expired(self):
        assert not is_expired(self._obj("not-a-date"))
