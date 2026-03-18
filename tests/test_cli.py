"""Tests for llmfs CLI commands."""
import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock

from llmfs.cli.main import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mem_dir(tmp_path):
    return str(tmp_path / "llmfs")


class TestCLIInit:
    def test_init_creates_dir(self, runner, tmp_path):
        target = tmp_path / "newllmfs"
        result = runner.invoke(cli, ["init", "--llmfs-path", str(target)])
        assert result.exit_code == 0
        assert target.exists()

    def test_init_already_exists(self, runner, tmp_path):
        target = tmp_path / "existing"
        target.mkdir()
        result = runner.invoke(cli, ["init", "--llmfs-path", str(target)])
        assert result.exit_code == 0
        assert "already" in result.output.lower()


class TestCLIWrite:
    def test_write_basic(self, runner, mem_dir):
        result = runner.invoke(
            cli, ["write", "/test/note", "hello world", "--llmfs-path", mem_dir]
        )
        assert result.exit_code == 0, result.output
        assert "Stored" in result.output

    def test_write_with_tags(self, runner, mem_dir):
        result = runner.invoke(
            cli, ["write", "/test/tagged", "content", "--tags", "a,b",
                  "--llmfs-path", mem_dir]
        )
        assert result.exit_code == 0, result.output

    def test_write_missing_content(self, runner, mem_dir):
        # Provide empty stdin
        result = runner.invoke(
            cli, ["write", "/test/empty", "--llmfs-path", mem_dir],
            input="",
        )
        assert result.exit_code != 0


class TestCLIRead:
    def test_read_existing(self, runner, mem_dir):
        runner.invoke(cli, ["write", "/k/r", "read me", "--llmfs-path", mem_dir])
        result = runner.invoke(cli, ["read", "/k/r", "--llmfs-path", mem_dir])
        assert result.exit_code == 0, result.output

    def test_read_missing(self, runner, mem_dir):
        # init the store first
        runner.invoke(cli, ["init", "--llmfs-path", mem_dir])
        result = runner.invoke(cli, ["read", "/missing", "--llmfs-path", mem_dir])
        assert result.exit_code != 0


class TestCLISearch:
    def test_search_returns_table(self, runner, mem_dir):
        runner.invoke(cli, ["write", "/k/x", "JWT authentication", "--llmfs-path", mem_dir])
        result = runner.invoke(cli, ["search", "authentication", "--llmfs-path", mem_dir])
        assert result.exit_code == 0, result.output

    def test_search_no_results(self, runner, mem_dir):
        runner.invoke(cli, ["init", "--llmfs-path", mem_dir])
        result = runner.invoke(cli, ["search", "nothing here", "--llmfs-path", mem_dir])
        assert result.exit_code == 0


class TestCLIStatus:
    def test_status_shows_info(self, runner, mem_dir):
        runner.invoke(cli, ["init", "--llmfs-path", mem_dir])
        result = runner.invoke(cli, ["status", "--llmfs-path", mem_dir])
        assert result.exit_code == 0
        assert "Total" in result.output or "LLMFS" in result.output


class TestCLIForget:
    def test_forget_with_yes_flag(self, runner, mem_dir):
        runner.invoke(cli, ["write", "/k/del", "bye", "--llmfs-path", mem_dir])
        result = runner.invoke(
            cli, ["forget", "/k/del", "--yes", "--llmfs-path", mem_dir]
        )
        assert result.exit_code == 0
        assert "Deleted" in result.output

    def test_forget_no_args(self, runner, mem_dir):
        runner.invoke(cli, ["init", "--llmfs-path", mem_dir])
        result = runner.invoke(cli, ["forget", "--llmfs-path", mem_dir])
        assert result.exit_code != 0


class TestCLIGC:
    def test_gc_runs(self, runner, mem_dir):
        runner.invoke(cli, ["init", "--llmfs-path", mem_dir])
        result = runner.invoke(cli, ["gc", "--llmfs-path", mem_dir])
        assert result.exit_code == 0
        assert "GC" in result.output
