"""
Tests for llmfs.context.extractor — ArtifactExtractor.
"""
import json
import tempfile
from pathlib import Path

import pytest
from llmfs import MemoryFS
from llmfs.context.extractor import ArtifactExtractor


@pytest.fixture
def mem(tmp_path):
    return MemoryFS(path=tmp_path / "llmfs")


@pytest.fixture
def extractor():
    return ArtifactExtractor()


SESSION = "testsess"


class TestCodeBlockExtraction:
    def test_python_block_stored(self, mem, extractor):
        turn = {"id": "1", "role": "assistant",
                "content": "Here is the fix:\n```python\nprint('hello')\n```"}
        paths = extractor.extract(turn, mem=mem, session_id=SESSION)
        assert f"/session/{SESSION}/code/turn_1_0" in paths

    def test_code_block_content_correct(self, mem, extractor):
        turn = {"id": "2", "role": "assistant",
                "content": "```javascript\nconsole.log('hi')\n```"}
        extractor.extract(turn, mem=mem, session_id=SESSION)
        obj = mem.read(f"/session/{SESSION}/code/turn_2_0")
        assert "console.log" in obj.content

    def test_multiple_code_blocks(self, mem, extractor):
        content = "First:\n```python\nfoo()\n```\nSecond:\n```bash\nls -la\n```"
        turn = {"id": "3", "role": "assistant", "content": content}
        paths = extractor.extract(turn, mem=mem, session_id=SESSION)
        assert f"/session/{SESSION}/code/turn_3_0" in paths
        assert f"/session/{SESSION}/code/turn_3_1" in paths

    def test_no_code_block_returns_empty(self, mem, extractor):
        turn = {"id": "4", "role": "user", "content": "Hello, how are you?"}
        paths = extractor.extract(turn, mem=mem, session_id=SESSION)
        code_paths = [p for p in paths if "/code/" in p]
        assert code_paths == []


class TestErrorExtraction:
    def test_traceback_extracted(self, mem, extractor):
        content = (
            "Got an error:\n"
            "Traceback (most recent call last):\n"
            "  File 'auth.py', line 45, in verify\n"
            "    token = tokens[user_id]\n"
            "KeyError: 'user_id'\n"
        )
        turn = {"id": "5", "role": "assistant", "content": content}
        paths = extractor.extract(turn, mem=mem, session_id=SESSION)
        assert f"/session/{SESSION}/errors/turn_5" in paths

    def test_error_colon_format(self, mem, extractor):
        content = "TypeError: 'NoneType' object is not subscriptable"
        turn = {"id": "6", "role": "assistant", "content": content}
        paths = extractor.extract(turn, mem=mem, session_id=SESSION)
        assert f"/session/{SESSION}/errors/turn_6" in paths

    def test_no_error_returns_empty(self, mem, extractor):
        turn = {"id": "7", "role": "user", "content": "Everything works fine!"}
        paths = extractor.extract(turn, mem=mem, session_id=SESSION)
        error_paths = [p for p in paths if "/errors/" in p]
        assert error_paths == []


class TestFileRefExtraction:
    def test_python_file_extracted(self, mem, extractor):
        turn = {"id": "8", "role": "assistant",
                "content": "The bug is in auth.py at line 45."}
        paths = extractor.extract(turn, mem=mem, session_id=SESSION)
        assert f"/session/{SESSION}/files/turn_8" in paths

    def test_file_refs_json_content(self, mem, extractor):
        turn = {"id": "9", "role": "assistant",
                "content": "Look at config.yaml and src/main.py for details."}
        paths = extractor.extract(turn, mem=mem, session_id=SESSION)
        file_path = f"/session/{SESSION}/files/turn_9"
        if file_path in paths:
            obj = mem.read(file_path)
            refs = json.loads(obj.content)
            assert isinstance(refs, list)
            assert any("main.py" in r for r in refs)

    def test_no_file_refs(self, mem, extractor):
        turn = {"id": "10", "role": "user", "content": "Please help me out."}
        paths = extractor.extract(turn, mem=mem, session_id=SESSION)
        file_paths = [p for p in paths if "/files/" in p]
        assert file_paths == []


class TestDecisionExtraction:
    def test_decided_keyword(self, mem, extractor):
        turn = {"id": "11", "role": "assistant",
                "content": "I decided to use PostgreSQL for the database."}
        paths = extractor.extract(turn, mem=mem, session_id=SESSION)
        assert f"/session/{SESSION}/decisions/turn_11" in paths

    def test_will_use_keyword(self, mem, extractor):
        turn = {"id": "12", "role": "assistant",
                "content": "We will use JWT for authentication going forward."}
        paths = extractor.extract(turn, mem=mem, session_id=SESSION)
        assert f"/session/{SESSION}/decisions/turn_12" in paths

    def test_no_decision_keywords(self, mem, extractor):
        turn = {"id": "13", "role": "user", "content": "Hello, how are you?"}
        paths = extractor.extract(turn, mem=mem, session_id=SESSION)
        decision_paths = [p for p in paths if "/decisions/" in p]
        assert decision_paths == []


class TestFullTurnArchive:
    def test_full_turn_stored(self, mem, extractor):
        turn = {"id": "14", "role": "user", "content": "This is a test turn."}
        path = extractor.extract_and_store_full_turn(turn, mem=mem, session_id=SESSION)
        assert path == f"/session/{SESSION}/turns/14"
        obj = mem.read(path)
        assert "test turn" in obj.content

    def test_full_turn_role_in_tags(self, mem, extractor):
        turn = {"id": "15", "role": "assistant", "content": "Reply content."}
        extractor.extract_and_store_full_turn(turn, mem=mem, session_id=SESSION)
        obj = mem.read(f"/session/{SESSION}/turns/15")
        assert "assistant" in obj.tags
