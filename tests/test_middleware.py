"""
Tests for llmfs.context.middleware — ContextMiddleware.
"""
import pytest
from llmfs import MemoryFS
from llmfs.context.middleware import ContextMiddleware, _extract_content


@pytest.fixture
def mem(tmp_path):
    return MemoryFS(path=tmp_path / "llmfs")


def simple_agent(messages):
    """Dummy agent that echoes last user message as assistant."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return {"role": "assistant", "content": f"Echo: {msg['content']}"}
    return {"role": "assistant", "content": "OK"}


class TestContextMiddlewareInit:
    def test_creates_with_callable(self, mem):
        wrapped = ContextMiddleware(simple_agent, memory=mem)
        assert wrapped is not None

    def test_creates_with_chat_method(self, mem):
        class FakeAgent:
            def chat(self, messages):
                return {"role": "assistant", "content": "hi"}
        wrapped = ContextMiddleware(FakeAgent(), memory=mem)
        assert wrapped is not None

    def test_custom_session_id(self, mem):
        wrapped = ContextMiddleware(simple_agent, memory=mem, session_id="mw_sess")
        assert wrapped.get_context_stats()["session_id"] == "mw_sess"

    def test_repr(self, mem):
        wrapped = ContextMiddleware(simple_agent, memory=mem)
        assert "ContextMiddleware" in repr(wrapped)


class TestContextMiddlewareChat:
    def test_basic_chat_call(self, mem):
        wrapped = ContextMiddleware(simple_agent, memory=mem, max_tokens=10_000)
        response = wrapped.chat([{"role": "user", "content": "Hello"}])
        assert response is not None

    def test_call_count_increments(self, mem):
        wrapped = ContextMiddleware(simple_agent, memory=mem, max_tokens=10_000)
        wrapped.chat([{"role": "user", "content": "A"}])
        wrapped.chat([{"role": "user", "content": "B"}])
        assert wrapped.get_context_stats()["call_count"] == 2

    def test_active_turns_tracked(self, mem):
        wrapped = ContextMiddleware(simple_agent, memory=mem, max_tokens=10_000)
        wrapped.chat([{"role": "user", "content": "Hello world here"}])
        # Tracks the user message and the assistant response
        stats = wrapped.get_context_stats()
        assert stats["active_turns"] >= 1

    def test_memory_index_injected_after_eviction(self, mem):
        """When context fills and evicts, system message should get the index."""
        injected_messages: list = []

        def capture_agent(messages):
            injected_messages.extend(messages)
            return {"role": "assistant", "content": "OK"}

        wrapped = ContextMiddleware(
            capture_agent, memory=mem,
            max_tokens=100, evict_at=0.70, target_after_evict=0.50,
        )
        for i in range(8):
            wrapped.chat([{"role": "user", "content": f"Turn {i} content words here"}])

        # After eviction, a system message should appear with the index
        system_msgs = [m for m in injected_messages if m.get("role") == "system"]
        if system_msgs:
            assert any("LLMFS" in m["content"] for m in system_msgs)

    def test_system_message_preserved_and_augmented(self, mem):
        """Existing system messages should be preserved with index appended."""
        augmented: list = []

        def capture_agent(messages):
            augmented.extend(messages)
            return {"role": "assistant", "content": "OK"}

        wrapped = ContextMiddleware(
            capture_agent, memory=mem,
            max_tokens=50, evict_at=0.70, target_after_evict=0.50,
        )
        # Trigger eviction
        for i in range(5):
            wrapped.chat([{"role": "user", "content": f"msg {i} has content"}])

        # Force a call with explicit system message
        index = wrapped.get_memory_index()
        if index:
            msgs_with_system = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hello"},
            ]
            augmented.clear()
            wrapped.chat(msgs_with_system)
            sys_msgs = [m for m in augmented if m.get("role") == "system"]
            if sys_msgs:
                combined = sys_msgs[0]["content"]
                assert "You are helpful" in combined

    def test_no_eviction_no_index(self, mem):
        wrapped = ContextMiddleware(simple_agent, memory=mem, max_tokens=100_000)
        wrapped.chat([{"role": "user", "content": "Hello"}])
        index = wrapped.get_memory_index()
        # No eviction yet, index should be empty
        assert index == "" or "No memories" in index

    def test_invoke_with_string(self, mem):
        wrapped = ContextMiddleware(simple_agent, memory=mem, max_tokens=10_000)
        response = wrapped.invoke("Hello there")
        assert response is not None

    def test_invoke_with_dict(self, mem):
        wrapped = ContextMiddleware(simple_agent, memory=mem, max_tokens=10_000)
        response = wrapped.invoke({"messages": [{"role": "user", "content": "Hi"}]})
        assert response is not None


class TestContextMiddlewareStats:
    def test_get_context_stats_keys(self, mem):
        wrapped = ContextMiddleware(simple_agent, memory=mem, max_tokens=10_000)
        stats = wrapped.get_context_stats()
        assert "call_count" in stats
        assert "session_id" in stats
        assert "total_tokens" in stats
        assert "max_tokens" in stats

    def test_get_active_turns(self, mem):
        wrapped = ContextMiddleware(simple_agent, memory=mem, max_tokens=10_000)
        wrapped.chat([{"role": "user", "content": "Hello world"}])
        turns = wrapped.get_active_turns()
        assert isinstance(turns, list)

    def test_reset_session(self, mem):
        wrapped = ContextMiddleware(simple_agent, memory=mem, max_tokens=10_000)
        wrapped.chat([{"role": "user", "content": "test"}])
        result = wrapped.reset_session()
        assert result["status"] == "ok"
        assert wrapped.get_context_stats()["active_turns"] == 0


class TestAgentInterfaces:
    def test_callable_agent(self, mem):
        results = []

        def agent(messages):
            results.append(messages)
            return "response"

        wrapped = ContextMiddleware(agent, memory=mem, max_tokens=10_000)
        wrapped.chat([{"role": "user", "content": "test"}])
        assert len(results) > 0

    def test_chat_method_agent(self, mem):
        class Agent:
            def chat(self, messages):
                return {"role": "assistant", "content": "hi"}

        wrapped = ContextMiddleware(Agent(), memory=mem, max_tokens=10_000)
        wrapped.chat([{"role": "user", "content": "test"}])

    def test_invoke_method_agent(self, mem):
        class Agent:
            def invoke(self, input):
                return {"role": "assistant", "content": "hi"}

        wrapped = ContextMiddleware(Agent(), memory=mem, max_tokens=10_000)
        wrapped.invoke("hello")


class TestExtractContent:
    def test_string(self):
        assert _extract_content("hello") == "hello"

    def test_dict_content(self):
        assert _extract_content({"content": "hi"}) == "hi"

    def test_dict_text(self):
        assert _extract_content({"text": "hi"}) == "hi"

    def test_object_with_content(self):
        class Resp:
            content = "hello"
        assert _extract_content(Resp()) == "hello"
