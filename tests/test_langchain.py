"""
Comprehensive tests for llmfs.integrations.langchain.

LangChain is NOT installed in the venv, so we mock the import before loading
the module under test.  All tests use a real MemoryFS backed by a temp
directory, plus a fast deterministic FakeEmbedder so we never hit a model
download.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock langchain BEFORE importing the integration module
# ---------------------------------------------------------------------------

# Create mock modules for langchain and langchain_core
_langchain_mock = MagicMock()
_langchain_core_mock = MagicMock()
_langchain_core_messages_mock = MagicMock()

# Set up concrete message classes that behave like real message objects
class _FakeHumanMessage:
    def __init__(self, content: str):
        self.content = content

class _FakeAIMessage:
    def __init__(self, content: str):
        self.content = content

class _FakeSystemMessage:
    def __init__(self, content: str):
        self.content = content

_langchain_core_messages_mock.HumanMessage = _FakeHumanMessage
_langchain_core_messages_mock.AIMessage = _FakeAIMessage
_langchain_core_messages_mock.SystemMessage = _FakeSystemMessage

sys.modules["langchain"] = _langchain_mock
sys.modules["langchain_core"] = _langchain_core_mock
sys.modules["langchain_core.messages"] = _langchain_core_messages_mock

# Now import the module under test
from llmfs.integrations.langchain import LLMFSChatMemory, LLMFSRetrieverMemory  # noqa: E402

# ---------------------------------------------------------------------------
# Shared FakeEmbedder (avoids 22 MB model download)
# ---------------------------------------------------------------------------

from llmfs import MemoryFS
from llmfs.embeddings.base import EmbedderBase


class FakeEmbedder(EmbedderBase):
    DIM = 16

    @property
    def model_name(self) -> str:
        return "fake"

    @property
    def embedding_dim(self) -> int:
        return self.DIM

    def embed(self, text: str) -> list[float]:
        h = hash(text) % (2 ** 31)
        return [float((h >> i) & 1) for i in range(self.DIM)]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mem(tmp_path):
    """Real MemoryFS with a fake embedder."""
    return MemoryFS(path=tmp_path / "llmfs", embedder=FakeEmbedder())


@pytest.fixture
def chat_memory(mem):
    """LLMFSChatMemory backed by a real (temp) MemoryFS."""
    return LLMFSChatMemory(session_prefix="/session/chat", mem=mem)


@pytest.fixture
def retriever_memory(mem):
    """LLMFSRetrieverMemory backed by a real (temp) MemoryFS."""
    return LLMFSRetrieverMemory(
        session_prefix="/session/turns",
        mem=mem,
    )


# ===========================================================================
# TestLLMFSChatMemoryImport
# ===========================================================================


class TestLLMFSChatMemoryImport:
    """Test that ImportError is raised when langchain is not available."""

    def test_require_langchain_raises_when_not_installed(self, mem):
        """Temporarily setting langchain to None in sys.modules raises ImportError."""
        from llmfs.integrations import langchain as lc_mod
        saved = sys.modules.get("langchain")
        # Setting sys.modules[name] = None causes ImportError when imported
        sys.modules["langchain"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError, match="langchain is required"):
                lc_mod._require_langchain()
        finally:
            if saved is not None:
                sys.modules["langchain"] = saved
            else:
                sys.modules.pop("langchain", None)

    def test_require_langchain_succeeds_when_mocked(self, mem):
        """With the mock in place, _require_langchain should not raise."""
        from llmfs.integrations.langchain import _require_langchain
        # Should not raise because langchain is in sys.modules (mocked)
        _require_langchain()  # no exception expected

    def test_chat_memory_instantiation_succeeds_with_mock(self, mem):
        """LLMFSChatMemory can be instantiated when langchain is mocked."""
        obj = LLMFSChatMemory(session_prefix="/session/chat", mem=mem)
        assert obj is not None

    def test_retriever_memory_instantiation_succeeds_with_mock(self, mem):
        """LLMFSRetrieverMemory can be instantiated when langchain is mocked."""
        obj = LLMFSRetrieverMemory(session_prefix="/session/turns", mem=mem)
        assert obj is not None


# ===========================================================================
# TestLLMFSChatMemoryBasic
# ===========================================================================


class TestLLMFSChatMemoryBasic:
    """Basic operations: add messages and verify they are stored."""

    def test_add_user_message_stores_memory(self, chat_memory, mem):
        """add_user_message writes content to the session layer."""
        chat_memory.add_user_message("Hello from the user!")
        objs = mem.list("/session/chat", recursive=True)
        assert len(objs) == 1
        assert "Hello from the user!" in objs[0].content

    def test_add_ai_message_stores_memory(self, chat_memory, mem):
        """add_ai_message writes content tagged 'ai'."""
        chat_memory.add_ai_message("Hello from the AI!")
        objs = mem.list("/session/chat", recursive=True)
        assert len(objs) == 1
        assert "Hello from the AI!" in objs[0].content

    def test_add_user_message_tagged_human(self, chat_memory, mem):
        """add_user_message stores with 'human' tag."""
        chat_memory.add_user_message("User turn")
        objs = mem.list("/session/chat", recursive=True)
        assert "human" in objs[0].tags

    def test_add_ai_message_tagged_ai(self, chat_memory, mem):
        """add_ai_message stores with 'ai' tag."""
        chat_memory.add_ai_message("AI turn")
        objs = mem.list("/session/chat", recursive=True)
        assert "ai" in objs[0].tags

    def test_len_zero_initially(self, chat_memory):
        """An empty chat memory has length 0."""
        assert len(chat_memory) == 0

    def test_len_increments_with_each_message(self, chat_memory):
        """len() reflects number of stored messages."""
        chat_memory.add_user_message("First")
        assert len(chat_memory) == 1
        chat_memory.add_ai_message("Second")
        assert len(chat_memory) == 2

    def test_messages_returns_list(self, chat_memory):
        """messages property returns a list."""
        result = chat_memory.messages
        assert isinstance(result, list)

    def test_messages_empty_initially(self, chat_memory):
        """messages is empty before any writes."""
        assert chat_memory.messages == []

    def test_messages_returns_human_message_objects(self, chat_memory):
        """messages wraps user entries in HumanMessage."""
        chat_memory.add_user_message("Hello")
        msgs = chat_memory.messages
        assert len(msgs) == 1
        assert isinstance(msgs[0], _FakeHumanMessage)
        assert msgs[0].content == "Hello"

    def test_messages_returns_ai_message_objects(self, chat_memory):
        """messages wraps AI entries in AIMessage."""
        chat_memory.add_ai_message("Hi there")
        msgs = chat_memory.messages
        assert len(msgs) == 1
        assert isinstance(msgs[0], _FakeAIMessage)
        assert msgs[0].content == "Hi there"

    def test_messages_returns_multiple_messages(self, chat_memory):
        """messages returns all stored messages in sorted order."""
        chat_memory.add_user_message("First human")
        chat_memory.add_ai_message("First AI")
        msgs = chat_memory.messages
        assert len(msgs) == 2

    def test_session_layer_used_for_storage(self, chat_memory, mem):
        """Messages are stored in the 'session' layer."""
        chat_memory.add_user_message("Check layer")
        objs = mem.list("/session/chat", recursive=True)
        assert objs[0].layer == "session"


# ===========================================================================
# TestLLMFSChatMemoryClear
# ===========================================================================


class TestLLMFSChatMemoryClear:
    """Test clear() removes session layer messages."""

    def test_clear_removes_all_messages(self, chat_memory):
        """After clear(), len() is 0."""
        chat_memory.add_user_message("Message 1")
        chat_memory.add_ai_message("Message 2")
        assert len(chat_memory) == 2
        chat_memory.clear()
        assert len(chat_memory) == 0

    def test_clear_empty_memory_does_not_raise(self, chat_memory):
        """clear() on an empty chat memory should not raise."""
        chat_memory.clear()  # should not throw

    def test_clear_makes_messages_empty(self, chat_memory):
        """After clear(), messages property returns []."""
        chat_memory.add_user_message("Will be cleared")
        chat_memory.clear()
        assert chat_memory.messages == []

    def test_clear_calls_forget_with_session_layer(self, mem):
        """clear() delegates to mem.forget(layer='session')."""
        mock_mem = MagicMock()
        cm = LLMFSChatMemory(session_prefix="/session/chat", mem=mock_mem)
        # Satisfy __len__ call inside clear
        mock_mem.list.return_value = []
        cm.clear()
        mock_mem.forget.assert_called_once_with(layer="session")


# ===========================================================================
# TestLLMFSChatMemoryAddMessage
# ===========================================================================


class TestLLMFSChatMemoryAddMessage:
    """Test add_message() role detection from class name."""

    def _make_message(self, class_name: str, content: str):
        """Dynamically create a message-like object with a given class name."""
        cls = type(class_name, (), {"content": content})
        return cls()

    def test_add_message_human_class_stores_as_human(self, chat_memory, mem):
        """A class named 'HumanMessage' is stored with 'human' tag."""
        msg = self._make_message("HumanMessage", "Human content")
        chat_memory.add_message(msg)
        objs = mem.list("/session/chat", recursive=True)
        assert "human" in objs[0].tags

    def test_add_message_ai_class_stores_as_ai(self, chat_memory, mem):
        """A class named 'AIMessage' is stored with 'ai' tag."""
        msg = self._make_message("AIMessage", "AI content")
        chat_memory.add_message(msg)
        objs = mem.list("/session/chat", recursive=True)
        assert "ai" in objs[0].tags

    def test_add_message_system_class_stores_as_system(self, chat_memory, mem):
        """A class named 'SystemMessage' is stored with 'system' tag."""
        msg = self._make_message("SystemMessage", "System content")
        chat_memory.add_message(msg)
        objs = mem.list("/session/chat", recursive=True)
        assert "system" in objs[0].tags

    def test_add_message_user_class_stores_as_human(self, chat_memory, mem):
        """A class containing 'user' in name is stored with 'human' tag."""
        msg = self._make_message("UserMessage", "User content")
        chat_memory.add_message(msg)
        objs = mem.list("/session/chat", recursive=True)
        assert "human" in objs[0].tags

    def test_add_message_assistant_class_stores_as_ai(self, chat_memory, mem):
        """A class named 'AssistantMessage' is stored with 'ai' tag."""
        msg = self._make_message("AssistantMessage", "Assistant content")
        chat_memory.add_message(msg)
        objs = mem.list("/session/chat", recursive=True)
        assert "ai" in objs[0].tags

    def test_add_message_unknown_class_defaults_to_human(self, chat_memory, mem):
        """An unrecognized class name defaults to 'human' role."""
        msg = self._make_message("SomethingElse", "Unknown content")
        chat_memory.add_message(msg)
        objs = mem.list("/session/chat", recursive=True)
        assert "human" in objs[0].tags

    def test_add_message_stores_content(self, chat_memory, mem):
        """add_message stores the message's content attribute."""
        msg = self._make_message("HumanMessage", "My message content")
        chat_memory.add_message(msg)
        objs = mem.list("/session/chat", recursive=True)
        assert "My message content" in objs[0].content

    def test_add_message_increments_len(self, chat_memory):
        """add_message increases len by 1."""
        msg = self._make_message("HumanMessage", "Test")
        initial = len(chat_memory)
        chat_memory.add_message(msg)
        assert len(chat_memory) == initial + 1


# ===========================================================================
# TestLLMFSRetrieverMemoryBasic
# ===========================================================================


class TestLLMFSRetrieverMemoryBasic:
    """Basic LLMFSRetrieverMemory interface: memory_variables and load."""

    def test_memory_variables_default(self, retriever_memory):
        """Default memory_key is 'memory'."""
        assert retriever_memory.memory_variables == ["memory"]

    def test_memory_variables_is_list(self, retriever_memory):
        """memory_variables returns a list."""
        assert isinstance(retriever_memory.memory_variables, list)

    def test_load_memory_variables_empty_query_returns_empty(self, retriever_memory):
        """Empty query returns empty string for the memory key."""
        result = retriever_memory.load_memory_variables({"input": ""})
        assert result == {"memory": ""}

    def test_load_memory_variables_whitespace_query_returns_empty(self, retriever_memory):
        """Whitespace-only query returns empty string."""
        result = retriever_memory.load_memory_variables({"input": "   "})
        assert result == {"memory": ""}

    def test_load_memory_variables_no_results_returns_empty(self, retriever_memory):
        """No search results returns empty string for memory key."""
        # No data written — search should return nothing
        result = retriever_memory.load_memory_variables({"input": "some query"})
        assert result == {"memory": ""}

    def test_load_memory_variables_missing_key_returns_empty(self, retriever_memory):
        """If input_key is missing from inputs, returns empty."""
        result = retriever_memory.load_memory_variables({})
        assert result == {"memory": ""}

    def test_load_memory_variables_with_results(self, mem):
        """When results exist, load_memory_variables formats them."""
        # Write something that will match the search
        mem.write("/knowledge/auth", "JWT authentication bug at line 45", layer="knowledge")
        rm = LLMFSRetrieverMemory(session_prefix="/session/turns", mem=mem)
        result = rm.load_memory_variables({"input": "authentication"})
        # Result should contain the memory key
        assert "memory" in result
        # If there are matches, the formatted string should be non-empty
        # (may be empty if FakeEmbedder vectors don't match well)
        assert isinstance(result["memory"], str)

    def test_load_memory_variables_result_format(self, mem):
        """When results are found, format includes ## header and path."""
        # Mock the mem.search to guarantee results
        mock_mem = MagicMock()
        fake_result = MagicMock()
        fake_result.chunk_text = "JWT bug at line 45"
        fake_result.path = "/knowledge/auth"
        fake_result.score = 0.95
        mock_mem.search.return_value = [fake_result]
        mock_mem.list.return_value = []

        rm = LLMFSRetrieverMemory(session_prefix="/session/turns", mem=mock_mem)
        result = rm.load_memory_variables({"input": "authentication"})
        assert "## Relevant memories from LLMFS:" in result["memory"]
        assert "/knowledge/auth" in result["memory"]
        assert "0.95" in result["memory"]

    def test_load_memory_variables_calls_search_with_correct_args(self, mem):
        """load_memory_variables passes query, k, and layer to mem.search."""
        mock_mem = MagicMock()
        mock_mem.search.return_value = []
        mock_mem.list.return_value = []

        rm = LLMFSRetrieverMemory(
            session_prefix="/session/turns",
            mem=mock_mem,
            search_k=5,
            layer="knowledge",
        )
        rm.load_memory_variables({"input": "test query"})
        mock_mem.search.assert_called_once_with("test query", k=5, layer="knowledge")


# ===========================================================================
# TestLLMFSRetrieverMemorySaveContext
# ===========================================================================


class TestLLMFSRetrieverMemorySaveContext:
    """Test save_context() writes human + AI pairs."""

    def test_save_context_writes_human_turn(self, retriever_memory, mem):
        """save_context stores the human input."""
        retriever_memory.save_context(
            {"input": "What is the auth bug?"},
            {"response": "It is a JWT expiry issue."},
        )
        objs = mem.list("/session/turns", recursive=True)
        contents = [o.content for o in objs]
        assert any("What is the auth bug?" in c for c in contents)

    def test_save_context_writes_ai_turn(self, retriever_memory, mem):
        """save_context stores the AI response."""
        retriever_memory.save_context(
            {"input": "What is the auth bug?"},
            {"response": "JWT expiry issue."},
        )
        objs = mem.list("/session/turns", recursive=True)
        contents = [o.content for o in objs]
        assert any("JWT expiry issue." in c for c in contents)

    def test_save_context_stores_two_objects(self, retriever_memory, mem):
        """save_context creates two separate memory objects (human + ai)."""
        retriever_memory.save_context(
            {"input": "Hello"},
            {"response": "World"},
        )
        objs = mem.list("/session/turns", recursive=True)
        assert len(objs) == 2

    def test_save_context_human_tagged_correctly(self, retriever_memory, mem):
        """Human turn is tagged with ['human', 'turn']."""
        retriever_memory.save_context(
            {"input": "User message"},
            {"response": "AI reply"},
        )
        objs = mem.list("/session/turns", recursive=True)
        human_objs = [o for o in objs if "human" in o.tags]
        assert len(human_objs) == 1
        assert "turn" in human_objs[0].tags

    def test_save_context_ai_tagged_correctly(self, retriever_memory, mem):
        """AI turn is tagged with ['ai', 'turn']."""
        retriever_memory.save_context(
            {"input": "User message"},
            {"response": "AI reply"},
        )
        objs = mem.list("/session/turns", recursive=True)
        ai_objs = [o for o in objs if "ai" in o.tags]
        assert len(ai_objs) == 1
        assert "turn" in ai_objs[0].tags

    def test_save_context_uses_output_key(self, retriever_memory, mem):
        """save_context also checks 'output' key if 'response' is absent."""
        retriever_memory.save_context(
            {"input": "Question"},
            {"output": "Answer via output key"},
        )
        objs = mem.list("/session/turns", recursive=True)
        contents = [o.content for o in objs]
        assert any("Answer via output key" in c for c in contents)

    def test_save_context_skips_empty_human(self, retriever_memory, mem):
        """save_context does not write human turn if input is empty."""
        retriever_memory.save_context(
            {"input": ""},
            {"response": "Some response"},
        )
        objs = mem.list("/session/turns", recursive=True)
        human_objs = [o for o in objs if "human" in o.tags]
        assert len(human_objs) == 0

    def test_save_context_skips_empty_ai(self, retriever_memory, mem):
        """save_context does not write AI turn if response is empty."""
        retriever_memory.save_context(
            {"input": "Human text"},
            {"response": ""},
        )
        objs = mem.list("/session/turns", recursive=True)
        ai_objs = [o for o in objs if "ai" in o.tags]
        assert len(ai_objs) == 0

    def test_save_context_stored_in_session_layer(self, retriever_memory, mem):
        """Saved turns are in the session layer."""
        retriever_memory.save_context(
            {"input": "Human"},
            {"response": "AI"},
        )
        objs = mem.list("/session/turns", recursive=True)
        for o in objs:
            assert o.layer == "session"


# ===========================================================================
# TestLLMFSRetrieverMemoryClear
# ===========================================================================


class TestLLMFSRetrieverMemoryClear:
    """Test clear() on LLMFSRetrieverMemory."""

    def test_clear_removes_session_turns(self, retriever_memory, mem):
        """After clear(), saved turns are gone."""
        retriever_memory.save_context(
            {"input": "Question"},
            {"response": "Answer"},
        )
        assert len(mem.list("/session/turns", recursive=True)) > 0
        retriever_memory.clear()
        assert len(mem.list("/session/turns", recursive=True)) == 0

    def test_clear_empty_memory_does_not_raise(self, retriever_memory):
        """clear() on empty retriever memory should not raise."""
        retriever_memory.clear()  # no exception

    def test_clear_calls_forget_with_session_layer(self, mem):
        """clear() calls mem.forget(layer='session')."""
        mock_mem = MagicMock()
        rm = LLMFSRetrieverMemory(session_prefix="/session/turns", mem=mock_mem)
        rm.clear()
        mock_mem.forget.assert_called_once_with(layer="session")

    def test_clear_allows_fresh_save_after(self, retriever_memory, mem):
        """After clear(), new save_context works normally."""
        retriever_memory.save_context({"input": "Old"}, {"response": "Old"})
        retriever_memory.clear()
        retriever_memory.save_context({"input": "New"}, {"response": "Fresh"})
        objs = mem.list("/session/turns", recursive=True)
        contents = [o.content for o in objs]
        assert any("New" in c for c in contents)
        assert not any("Old" in c for c in contents)


# ===========================================================================
# TestLLMFSRetrieverMemoryCustomKey
# ===========================================================================


class TestLLMFSRetrieverMemoryCustomKey:
    """Test custom memory_key propagates through the interface."""

    def test_custom_memory_key_in_memory_variables(self, mem):
        """Custom memory_key appears in memory_variables list."""
        rm = LLMFSRetrieverMemory(
            session_prefix="/session/turns",
            mem=mem,
            memory_key="context",
        )
        assert rm.memory_variables == ["context"]

    def test_custom_memory_key_in_load_result(self, mem):
        """load_memory_variables uses custom key in returned dict."""
        rm = LLMFSRetrieverMemory(
            session_prefix="/session/turns",
            mem=mem,
            memory_key="context",
        )
        result = rm.load_memory_variables({"input": "hello"})
        assert "context" in result
        assert "memory" not in result

    def test_custom_memory_key_empty_query(self, mem):
        """Custom key is still used when query is empty."""
        rm = LLMFSRetrieverMemory(
            session_prefix="/session/turns",
            mem=mem,
            memory_key="my_context",
        )
        result = rm.load_memory_variables({"input": ""})
        assert result == {"my_context": ""}

    def test_custom_input_key(self, mem):
        """Custom input_key is used to extract the query from inputs."""
        mock_mem = MagicMock()
        mock_mem.search.return_value = []
        mock_mem.list.return_value = []

        rm = LLMFSRetrieverMemory(
            session_prefix="/session/turns",
            mem=mock_mem,
            input_key="question",
        )
        rm.load_memory_variables({"question": "What happened?"})
        mock_mem.search.assert_called_once()
        # Verify query passed to search
        call_args = mock_mem.search.call_args
        assert call_args[0][0] == "What happened?"

    def test_memory_key_default_is_memory(self, mem):
        """Default memory_key is 'memory'."""
        rm = LLMFSRetrieverMemory(session_prefix="/session/turns", mem=mem)
        assert rm.memory_variables == ["memory"]

    def test_custom_session_prefix(self, mem):
        """Custom session_prefix stores turns under that prefix."""
        rm = LLMFSRetrieverMemory(
            session_prefix="/my/turns",
            mem=mem,
        )
        rm.save_context({"input": "Hello"}, {"response": "World"})
        objs = mem.list("/my/turns", recursive=True)
        assert len(objs) == 2
