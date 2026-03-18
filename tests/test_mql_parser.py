"""
Comprehensive tests for the MQL (Memory Query Language) parser.

Covers:
- tokenize() lexer
- MQLParser().parse() with all grammar constructs
- All AST node types
- MQLParseError on invalid input
"""

import pytest

from llmfs.core.exceptions import MQLParseError
from llmfs.query.parser import (
    AndCondition,
    DateCondition,
    MQLParser,
    OrCondition,
    RelatedToCondition,
    SelectStatement,
    SimilarCondition,
    TagCondition,
    Token,
    TokenType,
    TopicCondition,
    tokenize,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def token_types(query: str) -> list[str]:
    """Return the list of token types (excluding EOF) for a query string."""
    return [t.type for t in tokenize(query) if t.type != TokenType.EOF]


def token_values(query: str) -> list[str]:
    """Return the list of token values (excluding EOF) for a query string."""
    return [t.value for t in tokenize(query) if t.type != TokenType.EOF]


def parse(query: str) -> SelectStatement:
    """Convenience wrapper around MQLParser().parse()."""
    return MQLParser().parse(query)


# ══════════════════════════════════════════════════════════════════════════════
# TestTokenizer
# ══════════════════════════════════════════════════════════════════════════════

class TestTokenizer:
    """Unit tests for the tokenize() lexer."""

    # ── Basic structure ───────────────────────────────────────────────────────

    def test_empty_string_returns_only_eof(self):
        tokens = tokenize("")
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF
        assert tokens[0].value == ""

    def test_whitespace_only_returns_only_eof(self):
        tokens = tokenize("   \t\n  ")
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF

    def test_last_token_is_always_eof(self):
        tokens = tokenize("SELECT memory FROM /")
        assert tokens[-1].type == TokenType.EOF

    def test_eof_position_is_end_of_string(self):
        query = "SELECT"
        tokens = tokenize(query)
        assert tokens[-1].position == len(query)

    # ── Keywords (case-insensitive) ───────────────────────────────────────────

    def test_keyword_select_lowercase(self):
        assert token_types("select") == [TokenType.SELECT]

    def test_keyword_select_uppercase(self):
        assert token_types("SELECT") == [TokenType.SELECT]

    def test_keyword_select_mixedcase(self):
        assert token_types("Select") == [TokenType.SELECT]

    def test_all_keywords_recognised(self):
        keywords_and_types = [
            ("select",   TokenType.SELECT),
            ("memory",   TokenType.MEMORY),
            ("from",     TokenType.FROM),
            ("where",    TokenType.WHERE),
            ("and",      TokenType.AND),
            ("or",       TokenType.OR),
            ("similar",  TokenType.SIMILAR),
            ("to",       TokenType.TO),
            ("tag",      TokenType.TAG),
            ("date",     TokenType.DATE),
            ("created",  TokenType.CREATED),
            ("modified", TokenType.MODIFIED),
            ("topic",    TokenType.TOPIC),
            ("related",  TokenType.RELATED),
            ("within",   TokenType.WITHIN),
            ("limit",    TokenType.LIMIT),
            ("order",    TokenType.ORDER),
            ("by",       TokenType.BY),
            ("asc",      TokenType.ASC),
            ("desc",     TokenType.DESC),
            ("in",       TokenType.IN),
        ]
        for word, expected_type in keywords_and_types:
            toks = tokenize(word)
            assert toks[0].type == expected_type, f"keyword {word!r} should be {expected_type}"

    # ── Identifiers ───────────────────────────────────────────────────────────

    def test_unknown_word_is_ident(self):
        assert token_types("knowledge") == [TokenType.IDENT]

    def test_ident_with_underscores(self):
        toks = tokenize("my_var_123")
        assert toks[0].type == TokenType.IDENT
        assert toks[0].value == "my_var_123"

    # ── String literals ───────────────────────────────────────────────────────

    def test_double_quoted_string_type(self):
        assert token_types('"hello world"') == [TokenType.STRING]

    def test_double_quoted_string_strips_quotes(self):
        toks = tokenize('"hello world"')
        assert toks[0].value == "hello world"

    def test_single_quoted_string_strips_quotes(self):
        toks = tokenize("'auth bug'")
        assert toks[0].value == "auth bug"

    def test_empty_string_literal(self):
        toks = tokenize('""')
        assert toks[0].type == TokenType.STRING
        assert toks[0].value == ""

    def test_string_with_special_chars(self):
        toks = tokenize('"2026-01-01"')
        assert toks[0].type == TokenType.STRING
        assert toks[0].value == "2026-01-01"

    # ── Path tokens ───────────────────────────────────────────────────────────

    def test_slash_root_is_path(self):
        toks = tokenize("/knowledge")
        assert toks[0].type == TokenType.PATH
        assert toks[0].value == "/knowledge"

    def test_nested_path_is_single_path_token(self):
        toks = tokenize("/projects/auth")
        assert toks[0].type == TokenType.PATH
        assert toks[0].value == "/projects/auth"

    def test_bare_slash_is_path(self):
        # "/" followed by space/EOF is still a PATH token (regex /[^\s,()=!<>\"']*)
        toks = tokenize("/")
        assert toks[0].type == TokenType.PATH

    # ── Integer tokens ────────────────────────────────────────────────────────

    def test_integer_token(self):
        toks = tokenize("42")
        assert toks[0].type == TokenType.INTEGER
        assert toks[0].value == "42"

    def test_integer_zero(self):
        toks = tokenize("0")
        assert toks[0].type == TokenType.INTEGER
        assert toks[0].value == "0"

    # ── Operator tokens ───────────────────────────────────────────────────────

    def test_operators_are_tokenised_correctly(self):
        mapping = [
            ("=",  TokenType.EQ),
            ("!=", TokenType.NEQ),
            (">",  TokenType.GT),
            ("<",  TokenType.LT),
            (">=", TokenType.GTE),
            ("<=", TokenType.LTE),
            ("(",  TokenType.LPAREN),
            (")",  TokenType.RPAREN),
            (",",  TokenType.COMMA),
        ]
        for src, expected in mapping:
            toks = tokenize(src)
            assert toks[0].type == expected, f"{src!r} should be {expected}"

    def test_gte_takes_precedence_over_gt(self):
        """'>=' must be lexed as a single GTE token, not GT + EQ."""
        types = token_types(">=")
        assert types == [TokenType.GTE]

    def test_lte_takes_precedence_over_lt(self):
        types = token_types("<=")
        assert types == [TokenType.LTE]

    def test_neq_takes_precedence_over_not(self):
        types = token_types("!=")
        assert types == [TokenType.NEQ]

    # ── Token position ────────────────────────────────────────────────────────

    def test_token_position_is_byte_offset(self):
        tokens = tokenize('SELECT "hello"')
        select_tok = tokens[0]
        str_tok = tokens[1]
        assert select_tok.position == 0
        assert str_tok.position == 7  # after "SELECT "

    # ── Error handling ────────────────────────────────────────────────────────

    def test_unexpected_character_raises_parse_error(self):
        with pytest.raises(MQLParseError):
            tokenize("SELECT @ memory")

    def test_unexpected_character_hash_raises_parse_error(self):
        with pytest.raises(MQLParseError):
            tokenize("# comment")


# ══════════════════════════════════════════════════════════════════════════════
# TestParserBasic
# ══════════════════════════════════════════════════════════════════════════════

class TestParserBasic:
    """Tests for minimal SELECT … FROM … with no WHERE clause."""

    def test_select_from_path_token(self):
        stmt = parse("SELECT memory FROM /knowledge")
        assert stmt.path == "/knowledge"
        assert stmt.conditions is None
        assert stmt.limit is None
        assert stmt.order_by is None

    def test_select_from_root_slash(self):
        stmt = parse("SELECT memory FROM /")
        assert stmt.path == "/"

    def test_select_from_nested_path(self):
        stmt = parse("SELECT memory FROM /projects/auth")
        assert stmt.path == "/projects/auth"

    def test_select_from_quoted_path(self):
        stmt = parse('SELECT memory FROM "/knowledge"')
        assert stmt.path == "/knowledge"

    def test_returns_select_statement_type(self):
        stmt = parse("SELECT memory FROM /")
        assert isinstance(stmt, SelectStatement)

    def test_default_order_dir_is_desc(self):
        stmt = parse("SELECT memory FROM /")
        assert stmt.order_dir == "desc"

    def test_case_insensitive_keywords(self):
        stmt = parse("select memory from /knowledge")
        assert stmt.path == "/knowledge"
        assert stmt.conditions is None

    def test_mixed_case_keywords(self):
        stmt = parse("Select Memory From /knowledge")
        assert stmt.path == "/knowledge"


# ══════════════════════════════════════════════════════════════════════════════
# TestParserSimilar
# ══════════════════════════════════════════════════════════════════════════════

class TestParserSimilar:
    """Tests for SIMILAR TO condition."""

    def test_similar_to_basic(self):
        stmt = parse('SELECT memory FROM / WHERE SIMILAR TO "auth bug"')
        assert isinstance(stmt.conditions, SimilarCondition)
        assert stmt.conditions.query_str == "auth bug"

    def test_similar_to_multi_word(self):
        stmt = parse('SELECT memory FROM / WHERE SIMILAR TO "error handling in Python"')
        assert stmt.conditions.query_str == "error handling in Python"

    def test_similar_to_empty_string(self):
        stmt = parse('SELECT memory FROM / WHERE SIMILAR TO ""')
        assert isinstance(stmt.conditions, SimilarCondition)
        assert stmt.conditions.query_str == ""

    def test_similar_to_path_preserved(self):
        stmt = parse('SELECT memory FROM /knowledge WHERE SIMILAR TO "JWT bug"')
        assert stmt.path == "/knowledge"
        assert isinstance(stmt.conditions, SimilarCondition)

    def test_similar_to_single_quoted(self):
        stmt = parse("SELECT memory FROM / WHERE SIMILAR TO 'auth bug'")
        assert isinstance(stmt.conditions, SimilarCondition)
        assert stmt.conditions.query_str == "auth bug"


# ══════════════════════════════════════════════════════════════════════════════
# TestParserTag
# ══════════════════════════════════════════════════════════════════════════════

class TestParserTag:
    """Tests for TAG condition (=, !=, IN)."""

    def test_tag_equals(self):
        stmt = parse('SELECT memory FROM / WHERE TAG = "python"')
        cond = stmt.conditions
        assert isinstance(cond, TagCondition)
        assert cond.tag == "python"
        assert cond.op == "="

    def test_tag_not_equals(self):
        stmt = parse('SELECT memory FROM / WHERE TAG != "python"')
        cond = stmt.conditions
        assert isinstance(cond, TagCondition)
        assert cond.op == "!="
        assert cond.tag == "python"

    def test_tag_in_single_value(self):
        stmt = parse('SELECT memory FROM / WHERE TAG IN ("python")')
        cond = stmt.conditions
        assert isinstance(cond, TagCondition)
        assert cond.op == "in"
        assert cond.values == ["python"]

    def test_tag_in_multiple_values(self):
        stmt = parse('SELECT memory FROM / WHERE TAG IN ("python", "auth", "bug")')
        cond = stmt.conditions
        assert isinstance(cond, TagCondition)
        assert cond.op == "in"
        assert cond.values == ["python", "auth", "bug"]

    def test_tag_in_tag_field_is_empty_string(self):
        """For TAG IN, tag field is empty string (values list is used)."""
        stmt = parse('SELECT memory FROM / WHERE TAG IN ("a", "b")')
        assert stmt.conditions.tag == ""

    def test_tag_equals_values_list_populated(self):
        """For TAG =, values list also contains the matched value."""
        stmt = parse('SELECT memory FROM / WHERE TAG = "python"')
        assert stmt.conditions.values == ["python"]

    def test_tag_not_equals_values_list_populated(self):
        stmt = parse('SELECT memory FROM / WHERE TAG != "python"')
        assert stmt.conditions.values == ["python"]


# ══════════════════════════════════════════════════════════════════════════════
# TestParserDate
# ══════════════════════════════════════════════════════════════════════════════

class TestParserDate:
    """Tests for DATE, CREATED, MODIFIED conditions with all operators."""

    def test_date_greater_than(self):
        stmt = parse('SELECT memory FROM / WHERE DATE > "2026-01-01"')
        cond = stmt.conditions
        assert isinstance(cond, DateCondition)
        assert cond.field == "date"
        assert cond.op == ">"
        assert cond.value == "2026-01-01"

    def test_date_less_than(self):
        stmt = parse('SELECT memory FROM / WHERE DATE < "2026-06-01"')
        cond = stmt.conditions
        assert cond.field == "date"
        assert cond.op == "<"

    def test_date_greater_equal(self):
        stmt = parse('SELECT memory FROM / WHERE DATE >= "2026-01-01"')
        assert stmt.conditions.op == ">="

    def test_date_less_equal(self):
        stmt = parse('SELECT memory FROM / WHERE DATE <= "2026-12-31"')
        assert stmt.conditions.op == "<="

    def test_date_equal(self):
        stmt = parse('SELECT memory FROM / WHERE DATE = "2026-03-15"')
        assert stmt.conditions.op == "="

    def test_created_field(self):
        stmt = parse('SELECT memory FROM / WHERE CREATED >= "2026-01-01"')
        cond = stmt.conditions
        assert isinstance(cond, DateCondition)
        assert cond.field == "created"
        assert cond.op == ">="

    def test_modified_field(self):
        stmt = parse('SELECT memory FROM / WHERE MODIFIED < "2026-06-01"')
        cond = stmt.conditions
        assert isinstance(cond, DateCondition)
        assert cond.field == "modified"
        assert cond.op == "<"

    def test_date_value_preserved(self):
        stmt = parse('SELECT memory FROM / WHERE DATE > "2025-12-31"')
        assert stmt.conditions.value == "2025-12-31"


# ══════════════════════════════════════════════════════════════════════════════
# TestParserTopic
# ══════════════════════════════════════════════════════════════════════════════

class TestParserTopic:
    """Tests for TOPIC condition."""

    def test_topic_basic(self):
        stmt = parse('SELECT memory FROM / WHERE TOPIC "database"')
        assert isinstance(stmt.conditions, TopicCondition)
        assert stmt.conditions.topic_str == "database"

    def test_topic_multi_word(self):
        stmt = parse('SELECT memory FROM / WHERE TOPIC "machine learning"')
        assert stmt.conditions.topic_str == "machine learning"

    def test_topic_with_path(self):
        stmt = parse('SELECT memory FROM /knowledge WHERE TOPIC "authentication"')
        assert stmt.path == "/knowledge"
        assert isinstance(stmt.conditions, TopicCondition)


# ══════════════════════════════════════════════════════════════════════════════
# TestParserRelated
# ══════════════════════════════════════════════════════════════════════════════

class TestParserRelated:
    """Tests for RELATED TO condition with and without WITHIN."""

    def test_related_to_string_path(self):
        stmt = parse('SELECT memory FROM / WHERE RELATED TO "/projects/auth"')
        cond = stmt.conditions
        assert isinstance(cond, RelatedToCondition)
        assert cond.anchor_path == "/projects/auth"

    def test_related_to_default_depth_is_2(self):
        stmt = parse('SELECT memory FROM / WHERE RELATED TO "/projects/auth"')
        assert stmt.conditions.depth == 2

    def test_related_to_with_within(self):
        stmt = parse('SELECT memory FROM / WHERE RELATED TO "/projects/auth" WITHIN 3')
        cond = stmt.conditions
        assert isinstance(cond, RelatedToCondition)
        assert cond.anchor_path == "/projects/auth"
        assert cond.depth == 3

    def test_related_to_within_depth_1(self):
        stmt = parse('SELECT memory FROM / WHERE RELATED TO "/root" WITHIN 1')
        assert stmt.conditions.depth == 1

    def test_related_to_path_token(self):
        """RELATED TO can also accept a bare PATH token (no quotes)."""
        stmt = parse("SELECT memory FROM / WHERE RELATED TO /projects/auth")
        cond = stmt.conditions
        assert isinstance(cond, RelatedToCondition)
        assert cond.anchor_path == "/projects/auth"

    def test_related_to_path_token_with_within(self):
        stmt = parse("SELECT memory FROM / WHERE RELATED TO /knowledge WITHIN 5")
        assert stmt.conditions.anchor_path == "/knowledge"
        assert stmt.conditions.depth == 5


# ══════════════════════════════════════════════════════════════════════════════
# TestParserLogical
# ══════════════════════════════════════════════════════════════════════════════

class TestParserLogical:
    """Tests for AND, OR, parenthesised conditions, and chained operators."""

    def test_and_condition(self):
        stmt = parse(
            'SELECT memory FROM / WHERE TAG = "python" AND SIMILAR TO "error handling"'
        )
        cond = stmt.conditions
        assert isinstance(cond, AndCondition)
        assert isinstance(cond.left, TagCondition)
        assert isinstance(cond.right, SimilarCondition)

    def test_or_condition(self):
        stmt = parse(
            'SELECT memory FROM / WHERE TAG = "python" OR TAG = "ruby"'
        )
        cond = stmt.conditions
        assert isinstance(cond, OrCondition)
        assert isinstance(cond.left, TagCondition)
        assert isinstance(cond.right, TagCondition)

    def test_chained_and_is_left_associative(self):
        stmt = parse(
            'SELECT memory FROM / WHERE TOPIC "a" AND TOPIC "b" AND TOPIC "c"'
        )
        # Parsed as (TOPIC "a" AND TOPIC "b") AND TOPIC "c"
        outer = stmt.conditions
        assert isinstance(outer, AndCondition)
        assert isinstance(outer.left, AndCondition)
        assert isinstance(outer.right, TopicCondition)

    def test_chained_or_is_left_associative(self):
        stmt = parse(
            'SELECT memory FROM / WHERE TOPIC "a" OR TOPIC "b" OR TOPIC "c"'
        )
        outer = stmt.conditions
        assert isinstance(outer, OrCondition)
        assert isinstance(outer.left, OrCondition)
        assert isinstance(outer.right, TopicCondition)

    def test_parenthesised_or_inside_and(self):
        stmt = parse(
            'SELECT memory FROM / WHERE TAG = "python" AND (TOPIC "a" OR TOPIC "b")'
        )
        cond = stmt.conditions
        assert isinstance(cond, AndCondition)
        assert isinstance(cond.left, TagCondition)
        assert isinstance(cond.right, OrCondition)

    def test_parenthesised_condition_preserves_inner(self):
        stmt = parse(
            'SELECT memory FROM / WHERE (SIMILAR TO "auth bug")'
        )
        # Parens are transparent — result is SimilarCondition directly
        assert isinstance(stmt.conditions, SimilarCondition)
        assert stmt.conditions.query_str == "auth bug"

    def test_and_similar_and_date(self):
        stmt = parse(
            'SELECT memory FROM / WHERE SIMILAR TO "bug" AND DATE > "2026-01-01"'
        )
        cond = stmt.conditions
        assert isinstance(cond, AndCondition)
        assert isinstance(cond.left, SimilarCondition)
        assert isinstance(cond.right, DateCondition)

    def test_or_has_correct_sides(self):
        stmt = parse(
            'SELECT memory FROM / WHERE DATE > "2026-01-01" OR TAG = "urgent"'
        )
        cond = stmt.conditions
        assert isinstance(cond, OrCondition)
        assert isinstance(cond.left, DateCondition)
        assert isinstance(cond.right, TagCondition)


# ══════════════════════════════════════════════════════════════════════════════
# TestParserModifiers
# ══════════════════════════════════════════════════════════════════════════════

class TestParserModifiers:
    """Tests for ORDER BY, LIMIT, and combinations thereof."""

    def test_limit_sets_limit(self):
        stmt = parse("SELECT memory FROM / LIMIT 10")
        assert stmt.limit == 10

    def test_limit_zero(self):
        stmt = parse("SELECT memory FROM / LIMIT 0")
        assert stmt.limit == 0

    def test_order_by_date(self):
        stmt = parse("SELECT memory FROM / ORDER BY date")
        assert stmt.order_by == "date"
        assert stmt.order_dir == "desc"  # default

    def test_order_by_date_asc(self):
        stmt = parse("SELECT memory FROM / ORDER BY date ASC")
        assert stmt.order_by == "date"
        assert stmt.order_dir == "asc"

    def test_order_by_date_desc(self):
        stmt = parse("SELECT memory FROM / ORDER BY date DESC")
        assert stmt.order_by == "date"
        assert stmt.order_dir == "desc"

    def test_order_by_score(self):
        stmt = parse("SELECT memory FROM / ORDER BY score")
        assert stmt.order_by == "score"

    def test_order_by_created(self):
        stmt = parse("SELECT memory FROM / ORDER BY created DESC")
        assert stmt.order_by == "created"
        assert stmt.order_dir == "desc"

    def test_order_by_modified(self):
        stmt = parse("SELECT memory FROM / ORDER BY modified ASC")
        assert stmt.order_by == "modified"
        assert stmt.order_dir == "asc"

    def test_order_by_normalised_to_lowercase(self):
        stmt = parse("SELECT memory FROM / ORDER BY DATE ASC")
        assert stmt.order_by == "date"

    def test_order_and_limit_together(self):
        stmt = parse("SELECT memory FROM / ORDER BY date DESC LIMIT 5")
        assert stmt.order_by == "date"
        assert stmt.order_dir == "desc"
        assert stmt.limit == 5

    def test_where_order_limit_full_query(self):
        stmt = parse(
            'SELECT memory FROM / WHERE TOPIC "database" ORDER BY date DESC LIMIT 10'
        )
        assert isinstance(stmt.conditions, TopicCondition)
        assert stmt.order_by == "date"
        assert stmt.order_dir == "desc"
        assert stmt.limit == 10

    def test_no_order_by_means_none(self):
        stmt = parse("SELECT memory FROM /")
        assert stmt.order_by is None

    def test_no_limit_means_none(self):
        stmt = parse("SELECT memory FROM /")
        assert stmt.limit is None

    def test_limit_is_integer_not_string(self):
        stmt = parse("SELECT memory FROM / LIMIT 7")
        assert isinstance(stmt.limit, int)
        assert stmt.limit == 7

    def test_where_and_limit_no_order(self):
        stmt = parse('SELECT memory FROM / WHERE SIMILAR TO "bug" LIMIT 5')
        assert isinstance(stmt.conditions, SimilarCondition)
        assert stmt.limit == 5
        assert stmt.order_by is None


# ══════════════════════════════════════════════════════════════════════════════
# TestParserErrors
# ══════════════════════════════════════════════════════════════════════════════

class TestParserErrors:
    """Tests that MQLParseError is raised for invalid input."""

    def test_empty_string_raises(self):
        with pytest.raises(MQLParseError):
            parse("")

    def test_missing_select_raises(self):
        with pytest.raises(MQLParseError):
            parse("memory FROM /knowledge")

    def test_missing_memory_raises(self):
        with pytest.raises(MQLParseError):
            parse("SELECT FROM /knowledge")

    def test_missing_from_raises(self):
        with pytest.raises(MQLParseError):
            parse("SELECT memory /knowledge")

    def test_missing_path_after_from_raises(self):
        with pytest.raises(MQLParseError):
            parse("SELECT memory FROM WHERE SIMILAR TO \"x\"")

    def test_similar_missing_to_raises(self):
        with pytest.raises(MQLParseError):
            parse('SELECT memory FROM / WHERE SIMILAR "auth bug"')

    def test_similar_missing_string_raises(self):
        with pytest.raises(MQLParseError):
            parse("SELECT memory FROM / WHERE SIMILAR TO")

    def test_tag_missing_operator_raises(self):
        with pytest.raises(MQLParseError):
            parse('SELECT memory FROM / WHERE TAG "python"')

    def test_tag_in_missing_paren_raises(self):
        with pytest.raises(MQLParseError):
            parse('SELECT memory FROM / WHERE TAG IN "python"')

    def test_related_missing_to_raises(self):
        with pytest.raises(MQLParseError):
            parse('SELECT memory FROM / WHERE RELATED "/projects/auth"')

    def test_limit_missing_integer_raises(self):
        with pytest.raises(MQLParseError):
            parse("SELECT memory FROM / LIMIT")

    def test_order_missing_by_raises(self):
        with pytest.raises(MQLParseError):
            parse("SELECT memory FROM / ORDER date")

    def test_order_by_missing_field_raises(self):
        with pytest.raises(MQLParseError):
            parse("SELECT memory FROM / ORDER BY")

    def test_unrecognised_character_raises(self):
        with pytest.raises(MQLParseError):
            parse("SELECT memory FROM / WHERE @invalid")

    def test_mql_parse_error_is_llmfs_error(self):
        """MQLParseError should be a subclass of LLMFSError (via LLMFSError)."""
        from llmfs.core.exceptions import LLMFSError
        with pytest.raises(LLMFSError):
            parse("")

    def test_mql_parse_error_stores_position(self):
        """MQLParseError should expose a position attribute."""
        try:
            tokenize("SELECT @ memory")
        except MQLParseError as exc:
            assert exc.position >= 0
        else:
            pytest.fail("Expected MQLParseError")

    def test_trailing_garbage_raises(self):
        with pytest.raises(MQLParseError):
            parse('SELECT memory FROM / WHERE SIMILAR TO "x" GARBAGE')

    def test_date_missing_operator_raises(self):
        with pytest.raises(MQLParseError):
            parse('SELECT memory FROM / WHERE DATE "2026-01-01"')

    def test_date_missing_value_raises(self):
        with pytest.raises(MQLParseError):
            parse("SELECT memory FROM / WHERE DATE >")
