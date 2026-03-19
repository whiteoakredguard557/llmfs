"""
Comprehensive tests for llmfs.integrations.openai_tools.

Tests cover:
  - LLMFS_TOOLS: structure, completeness, JSON schema validity
  - LLMFSToolHandler.handle(): dict format and object format tool calls
  - LLMFSToolHandler.handle_batch(): batch dispatch
  - LLMFSToolHandler.tool_result_messages(): OpenAI-format messages
  - _extract_tool_call() and _extract_call_id() helpers
  - Invalid JSON arguments handling
  - Unknown tool name handling

Note: ``openai`` is NOT installed. All tests run without it.
"""
from __future__ import annotations

import json
import types
import pytest

# ── Module under test ─────────────────────────────────────────────────────────
from llmfs.integrations.openai_tools import (
    LLMFS_TOOLS,
    LLMFSToolHandler,
    _extract_call_id,
    _extract_tool_call,
)
from llmfs import MemoryFS

# ── Expected tool names ───────────────────────────────────────────────────────
EXPECTED_TOOL_NAMES = [
    "memory_write",
    "memory_search",
    "memory_read",
    "memory_update",
    "memory_forget",
    "memory_relate",
    "memory_list",
]


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture()
def mem(tmp_path):
    """Fresh MemoryFS backed by a temp directory."""
    return MemoryFS(path=tmp_path)


@pytest.fixture()
def handler(mem):
    """LLMFSToolHandler wrapping a fresh MemoryFS."""
    return LLMFSToolHandler(mem)


@pytest.fixture()
def mem_with_data(tmp_path):
    """MemoryFS pre-populated with a couple of memories."""
    m = MemoryFS(path=tmp_path)
    m.write("/knowledge/auth", content="JWT expires in 15 minutes")
    m.write("/knowledge/db", content="We use PostgreSQL 15")
    return m


@pytest.fixture()
def handler_with_data(mem_with_data):
    return LLMFSToolHandler(mem_with_data)


# ── Helper: build a mock OpenAI-SDK ToolCall object ──────────────────────────

def _make_tool_call_obj(name: str, arguments: str, call_id: str = "call_abc123"):
    """Return a simple namespace mimicking an OpenAI ToolCall SDK object."""
    fn = types.SimpleNamespace(name=name, arguments=arguments)
    return types.SimpleNamespace(id=call_id, function=fn)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TestLLMFSToolDefinitions
# ═══════════════════════════════════════════════════════════════════════════════

class TestLLMFSToolDefinitions:
    """Tests for the LLMFS_TOOLS list of OpenAI tool definitions."""

    def test_is_list(self):
        assert isinstance(LLMFS_TOOLS, list)

    def test_has_exactly_six_tools(self):
        assert len(LLMFS_TOOLS) == 7

    def test_all_expected_names_present(self):
        names = [t["function"]["name"] for t in LLMFS_TOOLS]
        assert sorted(names) == sorted(EXPECTED_TOOL_NAMES)

    @pytest.mark.parametrize("tool", LLMFS_TOOLS)
    def test_each_tool_has_type_function(self, tool):
        assert tool.get("type") == "function"

    @pytest.mark.parametrize("tool", LLMFS_TOOLS)
    def test_each_tool_has_function_key(self, tool):
        assert "function" in tool

    @pytest.mark.parametrize("tool", LLMFS_TOOLS)
    def test_each_tool_function_has_name(self, tool):
        fn = tool["function"]
        assert "name" in fn and isinstance(fn["name"], str) and fn["name"]

    @pytest.mark.parametrize("tool", LLMFS_TOOLS)
    def test_each_tool_function_has_description(self, tool):
        fn = tool["function"]
        assert "description" in fn and isinstance(fn["description"], str) and fn["description"]

    @pytest.mark.parametrize("tool", LLMFS_TOOLS)
    def test_each_tool_function_has_parameters(self, tool):
        fn = tool["function"]
        assert "parameters" in fn

    @pytest.mark.parametrize("tool", LLMFS_TOOLS)
    def test_parameters_is_object_type(self, tool):
        params = tool["function"]["parameters"]
        assert isinstance(params, dict)
        assert params.get("type") == "object"

    @pytest.mark.parametrize("tool", LLMFS_TOOLS)
    def test_parameters_has_properties(self, tool):
        params = tool["function"]["parameters"]
        assert "properties" in params
        assert isinstance(params["properties"], dict)

    # ── Required-field checks for specific tools ──────────────────────────────

    def _get_tool(self, name: str) -> dict:
        for t in LLMFS_TOOLS:
            if t["function"]["name"] == name:
                return t
        pytest.fail(f"Tool {name!r} not found in LLMFS_TOOLS")

    def test_memory_write_required_fields(self):
        tool = self._get_tool("memory_write")
        required = tool["function"]["parameters"].get("required", [])
        assert "path" in required
        assert "content" in required

    def test_memory_write_has_path_property(self):
        tool = self._get_tool("memory_write")
        props = tool["function"]["parameters"]["properties"]
        assert "path" in props

    def test_memory_write_has_content_property(self):
        tool = self._get_tool("memory_write")
        props = tool["function"]["parameters"]["properties"]
        assert "content" in props

    def test_memory_search_required_fields(self):
        tool = self._get_tool("memory_search")
        required = tool["function"]["parameters"].get("required", [])
        assert "query" in required

    def test_memory_search_has_query_property(self):
        tool = self._get_tool("memory_search")
        props = tool["function"]["parameters"]["properties"]
        assert "query" in props

    def test_memory_relate_required_fields(self):
        tool = self._get_tool("memory_relate")
        required = tool["function"]["parameters"].get("required", [])
        assert "source" in required
        assert "target" in required
        assert "relationship" in required

    def test_memory_relate_has_source_target_relationship(self):
        tool = self._get_tool("memory_relate")
        props = tool["function"]["parameters"]["properties"]
        for key in ("source", "target", "relationship"):
            assert key in props, f"Missing property {key!r} in memory_relate"

    def test_memory_read_required_path(self):
        tool = self._get_tool("memory_read")
        required = tool["function"]["parameters"].get("required", [])
        assert "path" in required

    def test_memory_update_required_path(self):
        tool = self._get_tool("memory_update")
        required = tool["function"]["parameters"].get("required", [])
        assert "path" in required

    def test_memory_forget_no_required_fields(self):
        """memory_forget has no 'required' key or it is empty."""
        tool = self._get_tool("memory_forget")
        params = tool["function"]["parameters"]
        # Either absent or empty list
        required = params.get("required", [])
        assert required == [] or required is None

    def test_tool_names_are_unique(self):
        names = [t["function"]["name"] for t in LLMFS_TOOLS]
        assert len(names) == len(set(names))

    def test_all_names_are_strings(self):
        for t in LLMFS_TOOLS:
            assert isinstance(t["function"]["name"], str)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TestLLMFSToolHandlerDictFormat
# ═══════════════════════════════════════════════════════════════════════════════

class TestLLMFSToolHandlerDictFormat:
    """handle() with dict-format tool calls."""

    def test_handle_memory_write_returns_string(self, handler):
        tc = {
            "id": "call_001",
            "function": {
                "name": "memory_write",
                "arguments": json.dumps({"path": "/test/hello", "content": "world"}),
            },
        }
        result = handler.handle(tc)
        assert isinstance(result, str)

    def test_handle_memory_write_returns_valid_json(self, handler):
        tc = {
            "id": "call_002",
            "function": {
                "name": "memory_write",
                "arguments": json.dumps({"path": "/test/foo", "content": "bar"}),
            },
        }
        result = handler.handle(tc)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_handle_memory_write_status_ok(self, handler):
        tc = {
            "id": "call_003",
            "function": {
                "name": "memory_write",
                "arguments": json.dumps({"path": "/test/status", "content": "check"}),
            },
        }
        result = json.loads(handler.handle(tc))
        assert result.get("status") == "ok"

    def test_handle_memory_write_contains_path(self, handler):
        tc = {
            "id": "call_004",
            "function": {
                "name": "memory_write",
                "arguments": json.dumps({"path": "/test/mypath", "content": "data"}),
            },
        }
        result = json.loads(handler.handle(tc))
        assert result.get("path") == "/test/mypath"

    def test_handle_memory_search_returns_string(self, handler_with_data):
        tc = {
            "id": "call_010",
            "function": {
                "name": "memory_search",
                "arguments": json.dumps({"query": "authentication"}),
            },
        }
        result = handler_with_data.handle(tc)
        assert isinstance(result, str)

    def test_handle_memory_search_returns_valid_json(self, handler_with_data):
        tc = {
            "id": "call_011",
            "function": {
                "name": "memory_search",
                "arguments": json.dumps({"query": "JWT"}),
            },
        }
        result = json.loads(handler_with_data.handle(tc))
        assert isinstance(result, dict)

    def test_handle_memory_search_status_ok(self, handler_with_data):
        tc = {
            "id": "call_012",
            "function": {
                "name": "memory_search",
                "arguments": json.dumps({"query": "database"}),
            },
        }
        result = json.loads(handler_with_data.handle(tc))
        assert result.get("status") == "ok"

    def test_handle_memory_search_has_count_and_results(self, handler_with_data):
        tc = {
            "id": "call_013",
            "function": {
                "name": "memory_search",
                "arguments": json.dumps({"query": "PostgreSQL"}),
            },
        }
        result = json.loads(handler_with_data.handle(tc))
        assert "count" in result
        assert "results" in result
        assert isinstance(result["results"], list)

    def test_handle_memory_read_returns_ok(self, handler_with_data):
        tc = {
            "id": "call_020",
            "function": {
                "name": "memory_read",
                "arguments": json.dumps({"path": "/knowledge/auth"}),
            },
        }
        result = json.loads(handler_with_data.handle(tc))
        assert result.get("status") == "ok"

    def test_handle_memory_read_returns_content(self, handler_with_data):
        tc = {
            "id": "call_021",
            "function": {
                "name": "memory_read",
                "arguments": json.dumps({"path": "/knowledge/db"}),
            },
        }
        result = json.loads(handler_with_data.handle(tc))
        assert "content" in result

    def test_handle_memory_read_returns_path(self, handler_with_data):
        tc = {
            "id": "call_022",
            "function": {
                "name": "memory_read",
                "arguments": json.dumps({"path": "/knowledge/auth"}),
            },
        }
        result = json.loads(handler_with_data.handle(tc))
        assert result.get("path") == "/knowledge/auth"

    def test_handle_memory_forget_returns_ok(self, handler_with_data):
        tc = {
            "id": "call_030",
            "function": {
                "name": "memory_forget",
                "arguments": json.dumps({"path": "/knowledge/db"}),
            },
        }
        result = json.loads(handler_with_data.handle(tc))
        assert result.get("status") == "ok"

    def test_handle_memory_update_returns_ok(self, handler_with_data):
        tc = {
            "id": "call_040",
            "function": {
                "name": "memory_update",
                "arguments": json.dumps({
                    "path": "/knowledge/auth",
                    "append": " Updated info.",
                }),
            },
        }
        result = json.loads(handler_with_data.handle(tc))
        assert result.get("status") == "ok"

    def test_handle_memory_relate_returns_ok(self, handler_with_data):
        tc = {
            "id": "call_050",
            "function": {
                "name": "memory_relate",
                "arguments": json.dumps({
                    "source": "/knowledge/auth",
                    "target": "/knowledge/db",
                    "relationship": "related_to",
                }),
            },
        }
        result = json.loads(handler_with_data.handle(tc))
        assert result.get("status") == "ok"

    def test_handle_memory_relate_returns_relationship_id(self, handler_with_data):
        tc = {
            "id": "call_051",
            "function": {
                "name": "memory_relate",
                "arguments": json.dumps({
                    "source": "/knowledge/auth",
                    "target": "/knowledge/db",
                    "relationship": "caused_by",
                }),
            },
        }
        result = json.loads(handler_with_data.handle(tc))
        assert "relationship_id" in result


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TestLLMFSToolHandlerObjectFormat
# ═══════════════════════════════════════════════════════════════════════════════

class TestLLMFSToolHandlerObjectFormat:
    """handle() with mock SDK object format (has .function.name, .function.arguments, .id)."""

    def test_handle_write_obj_format_returns_string(self, handler):
        tc = _make_tool_call_obj(
            "memory_write",
            json.dumps({"path": "/obj/test", "content": "object format test"}),
            call_id="call_obj_001",
        )
        result = handler.handle(tc)
        assert isinstance(result, str)

    def test_handle_write_obj_format_valid_json(self, handler):
        tc = _make_tool_call_obj(
            "memory_write",
            json.dumps({"path": "/obj/test2", "content": "another test"}),
        )
        result = handler.handle(tc)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_handle_write_obj_format_status_ok(self, handler):
        tc = _make_tool_call_obj(
            "memory_write",
            json.dumps({"path": "/obj/status", "content": "ok check"}),
        )
        result = json.loads(handler.handle(tc))
        assert result.get("status") == "ok"

    def test_handle_search_obj_format(self, handler_with_data):
        tc = _make_tool_call_obj(
            "memory_search",
            json.dumps({"query": "JWT authentication"}),
            call_id="call_obj_010",
        )
        result = json.loads(handler_with_data.handle(tc))
        assert result.get("status") == "ok"
        assert "results" in result

    def test_handle_read_obj_format(self, handler_with_data):
        tc = _make_tool_call_obj(
            "memory_read",
            json.dumps({"path": "/knowledge/auth"}),
            call_id="call_obj_020",
        )
        result = json.loads(handler_with_data.handle(tc))
        assert result.get("status") == "ok"
        assert result.get("path") == "/knowledge/auth"

    def test_handle_obj_preserves_call_id_in_batch(self, handler):
        """Verify object format tool calls integrate properly with handle_batch."""
        tcs = [
            _make_tool_call_obj(
                "memory_write",
                json.dumps({"path": f"/obj/batch/{i}", "content": f"item {i}"}),
                call_id=f"call_batch_{i}",
            )
            for i in range(3)
        ]
        results = handler.handle_batch(tcs)
        assert len(results) == 3
        for r in results:
            parsed = json.loads(r)
            assert parsed.get("status") == "ok"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TestLLMFSToolHandlerBatch
# ═══════════════════════════════════════════════════════════════════════════════

class TestLLMFSToolHandlerBatch:
    """handle_batch() tests."""

    def test_handle_batch_returns_list(self, handler):
        tcs = [
            {"id": "c1", "function": {"name": "memory_write",
                                      "arguments": json.dumps({"path": "/b/1", "content": "a"})}},
            {"id": "c2", "function": {"name": "memory_write",
                                      "arguments": json.dumps({"path": "/b/2", "content": "b"})}},
        ]
        result = handler.handle_batch(tcs)
        assert isinstance(result, list)

    def test_handle_batch_same_length(self, handler):
        tcs = [
            {"id": f"c{i}", "function": {"name": "memory_write",
                                          "arguments": json.dumps({"path": f"/batch/{i}", "content": str(i)})}}
            for i in range(5)
        ]
        results = handler.handle_batch(tcs)
        assert len(results) == 5

    def test_handle_batch_each_element_is_string(self, handler):
        tcs = [
            {"id": "cx", "function": {"name": "memory_write",
                                      "arguments": json.dumps({"path": "/batch/x", "content": "x"})}},
        ]
        results = handler.handle_batch(tcs)
        for r in results:
            assert isinstance(r, str)

    def test_handle_batch_each_element_valid_json(self, handler):
        tcs = [
            {"id": f"d{i}", "function": {"name": "memory_write",
                                          "arguments": json.dumps({"path": f"/batch/d{i}", "content": f"val{i}"})}}
            for i in range(3)
        ]
        results = handler.handle_batch(tcs)
        for r in results:
            parsed = json.loads(r)
            assert isinstance(parsed, dict)

    def test_handle_batch_empty_list(self, handler):
        results = handler.handle_batch([])
        assert results == []

    def test_handle_batch_mixed_tools(self, mem_with_data):
        h = LLMFSToolHandler(mem_with_data)
        tcs = [
            {"id": "m1", "function": {"name": "memory_search",
                                      "arguments": json.dumps({"query": "JWT"})}},
            {"id": "m2", "function": {"name": "memory_read",
                                      "arguments": json.dumps({"path": "/knowledge/auth"})}},
        ]
        results = h.handle_batch(tcs)
        assert len(results) == 2
        for r in results:
            parsed = json.loads(r)
            assert parsed.get("status") == "ok"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TestToolResultMessages
# ═══════════════════════════════════════════════════════════════════════════════

class TestToolResultMessages:
    """tool_result_messages() returns proper OpenAI-format tool messages."""

    def _make_write_tc(self, idx: int, path_suffix: str = "") -> dict:
        return {
            "id": f"call_msg_{idx}",
            "function": {
                "name": "memory_write",
                "arguments": json.dumps({"path": f"/msg{path_suffix}/{idx}", "content": f"content{idx}"}),
            },
        }

    def test_returns_list(self, handler):
        tcs = [self._make_write_tc(0, "/a")]
        msgs = handler.tool_result_messages(tcs)
        assert isinstance(msgs, list)

    def test_returns_same_length(self, handler):
        tcs = [self._make_write_tc(i, "/b") for i in range(3)]
        msgs = handler.tool_result_messages(tcs)
        assert len(msgs) == 3

    def test_each_message_has_role_tool(self, handler):
        tcs = [self._make_write_tc(0, "/c")]
        msgs = handler.tool_result_messages(tcs)
        assert msgs[0]["role"] == "tool"

    def test_each_message_has_tool_call_id(self, handler):
        tcs = [self._make_write_tc(0, "/d")]
        msgs = handler.tool_result_messages(tcs)
        assert "tool_call_id" in msgs[0]

    def test_tool_call_id_matches_input_id(self, handler):
        tc = {"id": "specific_id_xyz", "function": {"name": "memory_write",
                                                      "arguments": json.dumps({"path": "/msg/e/0", "content": "e"})}}
        msgs = handler.tool_result_messages([tc])
        assert msgs[0]["tool_call_id"] == "specific_id_xyz"

    def test_each_message_has_content(self, handler):
        tcs = [self._make_write_tc(0, "/f")]
        msgs = handler.tool_result_messages(tcs)
        assert "content" in msgs[0]

    def test_content_is_valid_json_string(self, handler):
        tcs = [self._make_write_tc(0, "/g")]
        msgs = handler.tool_result_messages(tcs)
        content = msgs[0]["content"]
        assert isinstance(content, str)
        parsed = json.loads(content)
        assert isinstance(parsed, dict)

    def test_tool_result_messages_with_obj_format(self, handler):
        tc = _make_tool_call_obj(
            "memory_write",
            json.dumps({"path": "/msg/obj/1", "content": "obj msg"}),
            call_id="call_obj_msg_001",
        )
        msgs = handler.tool_result_messages([tc])
        assert len(msgs) == 1
        assert msgs[0]["role"] == "tool"
        assert msgs[0]["tool_call_id"] == "call_obj_msg_001"

    def test_tool_result_messages_empty(self, handler):
        msgs = handler.tool_result_messages([])
        assert msgs == []

    def test_multiple_messages_have_correct_ids(self, handler):
        tcs = [
            {"id": f"id_{i}", "function": {"name": "memory_write",
                                            "arguments": json.dumps({"path": f"/msg/h/{i}", "content": str(i)})}}
            for i in range(4)
        ]
        msgs = handler.tool_result_messages(tcs)
        for i, msg in enumerate(msgs):
            assert msg["tool_call_id"] == f"id_{i}"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TestHandleInvalidJSON
# ═══════════════════════════════════════════════════════════════════════════════

class TestHandleInvalidJSON:
    """Bad arguments JSON → error result, no exception raised."""

    def test_invalid_json_returns_string(self, handler):
        tc = {"id": "bad_001", "function": {"name": "memory_write", "arguments": "{not valid json"}}
        result = handler.handle(tc)
        assert isinstance(result, str)

    def test_invalid_json_returns_valid_json(self, handler):
        tc = {"id": "bad_002", "function": {"name": "memory_write", "arguments": "NOTJSON"}}
        result = handler.handle(tc)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_invalid_json_returns_error_status(self, handler):
        tc = {"id": "bad_003", "function": {"name": "memory_write", "arguments": "{"}}
        result = json.loads(handler.handle(tc))
        assert result.get("status") == "error"

    def test_invalid_json_error_message_mentions_json(self, handler):
        tc = {"id": "bad_004", "function": {"name": "memory_write", "arguments": "!!!"}}
        result = json.loads(handler.handle(tc))
        # error field should be a string mentioning JSON
        assert "error" in result
        assert isinstance(result["error"], str)

    def test_invalid_json_obj_format_returns_error(self, handler):
        tc = _make_tool_call_obj("memory_write", "not_valid{{{", call_id="bad_obj_001")
        result = json.loads(handler.handle(tc))
        assert result.get("status") == "error"

    def test_handle_batch_with_bad_json_does_not_raise(self, handler):
        tcs = [
            {"id": "bad_b1", "function": {"name": "memory_write", "arguments": "bad"}},
            {"id": "bad_b2", "function": {"name": "memory_write",
                                           "arguments": json.dumps({"path": "/ok/1", "content": "good"})}},
        ]
        results = handler.handle_batch(tcs)
        assert len(results) == 2
        bad = json.loads(results[0])
        good = json.loads(results[1])
        assert bad.get("status") == "error"
        assert good.get("status") == "ok"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. TestHandleUnknownTool
# ═══════════════════════════════════════════════════════════════════════════════

class TestHandleUnknownTool:
    """Unknown tool name → graceful error result, no crash."""

    def test_unknown_tool_returns_string(self, handler):
        tc = {"id": "unk_001", "function": {"name": "memory_teleport", "arguments": "{}"}}
        result = handler.handle(tc)
        assert isinstance(result, str)

    def test_unknown_tool_returns_valid_json(self, handler):
        tc = {"id": "unk_002", "function": {"name": "no_such_tool", "arguments": "{}"}}
        result = handler.handle(tc)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_unknown_tool_returns_error_status(self, handler):
        tc = {"id": "unk_003", "function": {"name": "memory_fly", "arguments": "{}"}}
        result = json.loads(handler.handle(tc))
        assert result.get("status") == "error"

    def test_unknown_tool_error_contains_tool_name(self, handler):
        tc = {"id": "unk_004", "function": {"name": "super_memory_zap", "arguments": "{}"}}
        result = json.loads(handler.handle(tc))
        assert "error" in result
        assert "super_memory_zap" in result["error"]

    def test_empty_tool_name_returns_error(self, handler):
        tc = {"id": "unk_005", "function": {"name": "", "arguments": "{}"}}
        result = json.loads(handler.handle(tc))
        assert result.get("status") == "error"

    def test_unknown_tool_obj_format_returns_error(self, handler):
        tc = _make_tool_call_obj("totally_fake_tool", "{}", call_id="unk_obj_001")
        result = json.loads(handler.handle(tc))
        assert result.get("status") == "error"


# ═══════════════════════════════════════════════════════════════════════════════
# 8. TestExtractHelpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractHelpers:
    """Unit tests for _extract_tool_call() and _extract_call_id()."""

    # ── _extract_tool_call ────────────────────────────────────────────────────

    def test_extract_tool_call_dict_name(self):
        tc = {"function": {"name": "memory_write", "arguments": '{"path":"/x"}'}}
        name, args = _extract_tool_call(tc)
        assert name == "memory_write"

    def test_extract_tool_call_dict_arguments(self):
        tc = {"function": {"name": "memory_search", "arguments": '{"query":"test"}'}}
        name, args = _extract_tool_call(tc)
        assert args == '{"query":"test"}'

    def test_extract_tool_call_dict_missing_function(self):
        tc = {}
        name, args = _extract_tool_call(tc)
        assert name == ""
        assert args == "{}"

    def test_extract_tool_call_dict_missing_arguments(self):
        tc = {"function": {"name": "memory_read"}}
        name, args = _extract_tool_call(tc)
        assert name == "memory_read"
        assert args == "{}"

    def test_extract_tool_call_obj_name(self):
        tc = _make_tool_call_obj("memory_search", '{"query":"hello"}', "call_x")
        name, args = _extract_tool_call(tc)
        assert name == "memory_search"

    def test_extract_tool_call_obj_arguments(self):
        tc = _make_tool_call_obj("memory_read", '{"path":"/knowledge/auth"}', "call_y")
        name, args = _extract_tool_call(tc)
        assert args == '{"path":"/knowledge/auth"}'

    def test_extract_tool_call_obj_no_function_attr(self):
        tc = types.SimpleNamespace(id="cz", name="memory_forget", arguments='{"path":"/x"}')
        name, args = _extract_tool_call(tc)
        assert name == "memory_forget"
        assert args == '{"path":"/x"}'

    # ── _extract_call_id ──────────────────────────────────────────────────────

    def test_extract_call_id_dict(self):
        tc = {"id": "call_12345", "function": {"name": "memory_write", "arguments": "{}"}}
        assert _extract_call_id(tc) == "call_12345"

    def test_extract_call_id_dict_missing(self):
        tc = {"function": {"name": "memory_write", "arguments": "{}"}}
        assert _extract_call_id(tc) == ""

    def test_extract_call_id_obj(self):
        tc = _make_tool_call_obj("memory_write", "{}", call_id="specific_id_99")
        assert _extract_call_id(tc) == "specific_id_99"

    def test_extract_call_id_obj_missing(self):
        tc = types.SimpleNamespace(function=types.SimpleNamespace(name="x", arguments="{}"))
        result = _extract_call_id(tc)
        assert result == ""
