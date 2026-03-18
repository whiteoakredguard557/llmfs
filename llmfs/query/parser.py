"""
MQL — Memory Query Language parser.

MQL is a small SQL-inspired query language for LLMFS.  It lets users and
agents express structured memory queries using natural, readable syntax.

Grammar summary::

    statement  ::= SELECT MEMORY FROM path_expr [WHERE condition] [LIMIT n] [ORDER BY field [ASC|DESC]]
    path_expr  ::= string_literal | '/'  | identifier ('/' identifier)*
    condition  ::= simple_cond  (AND | OR  simple_cond)*
    simple_cond::= similar_cond | tag_cond | date_cond | topic_cond | related_cond
    similar_cond  ::= SIMILAR TO string_literal
    tag_cond      ::= TAG (= | != | IN) string_literal | '(' string_list ')'
    date_cond     ::= DATE (> | < | >= | <= | =) string_literal
                    | CREATED (> | < | >= | <= | =) string_literal
                    | MODIFIED (> | < | >= | <= | =) string_literal
    topic_cond    ::= TOPIC string_literal
    related_cond  ::= RELATED TO string_literal [WITHIN n]

Examples::

    SELECT memory FROM /knowledge WHERE SIMILAR TO "auth bug"
    SELECT memory FROM / WHERE TAG = "python" AND SIMILAR TO "error handling" LIMIT 5
    SELECT memory FROM /events WHERE DATE > "2026-01-01"
    SELECT memory FROM /knowledge WHERE RELATED TO "/projects/auth" WITHIN 2
    SELECT memory FROM / WHERE TOPIC "database" ORDER BY date DESC LIMIT 10

Example::

    from llmfs.query.parser import MQLParser, SelectStatement

    parser = MQLParser()
    stmt = parser.parse('SELECT memory FROM /knowledge WHERE SIMILAR TO "JWT bug"')
    print(stmt.path)        # "/knowledge"
    print(stmt.limit)       # None
    print(stmt.conditions)  # [SimilarCondition(query_str='JWT bug')]
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from llmfs.core.exceptions import MQLParseError

__all__ = [
    "Token",
    "TokenType",
    "SelectStatement",
    "SimilarCondition",
    "TagCondition",
    "DateCondition",
    "TopicCondition",
    "RelatedToCondition",
    "AndCondition",
    "OrCondition",
    "MQLParser",
]


# ── Token types ───────────────────────────────────────────────────────────────

class TokenType:
    SELECT   = "SELECT"
    MEMORY   = "MEMORY"
    FROM     = "FROM"
    WHERE    = "WHERE"
    AND      = "AND"
    OR       = "OR"
    SIMILAR  = "SIMILAR"
    TO       = "TO"
    TAG      = "TAG"
    DATE     = "DATE"
    CREATED  = "CREATED"
    MODIFIED = "MODIFIED"
    TOPIC    = "TOPIC"
    RELATED  = "RELATED"
    WITHIN   = "WITHIN"
    LIMIT    = "LIMIT"
    ORDER    = "ORDER"
    BY       = "BY"
    ASC      = "ASC"
    DESC     = "DESC"
    IN       = "IN"
    # Punctuation / operators
    EQ       = "="
    NEQ      = "!="
    GT       = ">"
    LT       = "<"
    GTE      = ">="
    LTE      = "<="
    LPAREN   = "("
    RPAREN   = ")"
    COMMA    = ","
    SLASH    = "/"
    # Literals
    STRING   = "STRING"
    INTEGER  = "INTEGER"
    PATH     = "PATH"
    IDENT    = "IDENT"
    # Control
    EOF      = "EOF"


@dataclass
class Token:
    """A single lexer token.

    Attributes:
        type: Token type string (one of :class:`TokenType` constants).
        value: Raw string value from the source.
        position: Byte offset in the original query string.
    """
    type: str
    value: str
    position: int = 0

    def __repr__(self) -> str:
        return f"Token({self.type}, {self.value!r}, @{self.position})"


# ── AST node classes ──────────────────────────────────────────────────────────

@dataclass
class SimilarCondition:
    """``SIMILAR TO "<query>"`` — semantic similarity search."""
    query_str: str


@dataclass
class TagCondition:
    """``TAG = "python"`` or ``TAG IN ("a", "b")``."""
    tag: str
    op: str = "="          # "=", "!=", "in"
    values: list[str] = field(default_factory=list)  # for IN operator


@dataclass
class DateCondition:
    """``DATE > "2026-01-01"`` / ``CREATED >= "..."`` / ``MODIFIED < "..."``."""
    field: str             # "date" | "created" | "modified"
    op: str                # ">", "<", ">=", "<=", "="
    value: str


@dataclass
class TopicCondition:
    """``TOPIC "database"`` — keyword/topic filter."""
    topic_str: str


@dataclass
class RelatedToCondition:
    """``RELATED TO "/path" WITHIN 2`` — graph traversal."""
    anchor_path: str
    depth: int = 2


@dataclass
class AndCondition:
    """``left AND right``."""
    left: Any
    right: Any


@dataclass
class OrCondition:
    """``left OR right``."""
    left: Any
    right: Any


@dataclass
class SelectStatement:
    """Top-level MQL SELECT statement.

    Attributes:
        path: LLMFS path prefix to search under (e.g. ``"/knowledge"``).
        conditions: Root condition node (may be ``None`` for no WHERE clause).
        limit: Maximum results (``None`` = no limit).
        order_by: Field to sort by (``"date"``, ``"score"``).
        order_dir: ``"asc"`` or ``"desc"``.
    """
    path: str
    conditions: Any = None
    limit: int | None = None
    order_by: str | None = None
    order_dir: str = "desc"


# ── Lexer ─────────────────────────────────────────────────────────────────────

# Ordered token patterns — longer operators first
_TOKEN_PATTERNS: list[tuple[str, str]] = [
    (TokenType.STRING,   r'"[^"]*"|\'[^\']*\''),
    (TokenType.GTE,      r">="),
    (TokenType.LTE,      r"<="),
    (TokenType.NEQ,      r"!="),
    (TokenType.GT,       r">"),
    (TokenType.LT,       r"<"),
    (TokenType.EQ,       r"="),
    (TokenType.LPAREN,   r"\("),
    (TokenType.RPAREN,   r"\)"),
    (TokenType.COMMA,    r","),
    (TokenType.INTEGER,  r"\d+"),
    # Identifiers / keywords / paths — greedy path first
    (TokenType.PATH,     r"/[^\s,()=!<>\"']*"),
    (TokenType.IDENT,    r"[A-Za-z_][A-Za-z0-9_]*"),
]

_KEYWORDS: dict[str, str] = {
    "select":   TokenType.SELECT,
    "memory":   TokenType.MEMORY,
    "from":     TokenType.FROM,
    "where":    TokenType.WHERE,
    "and":      TokenType.AND,
    "or":       TokenType.OR,
    "similar":  TokenType.SIMILAR,
    "to":       TokenType.TO,
    "tag":      TokenType.TAG,
    "date":     TokenType.DATE,
    "created":  TokenType.CREATED,
    "modified": TokenType.MODIFIED,
    "topic":    TokenType.TOPIC,
    "related":  TokenType.RELATED,
    "within":   TokenType.WITHIN,
    "limit":    TokenType.LIMIT,
    "order":    TokenType.ORDER,
    "by":       TokenType.BY,
    "asc":      TokenType.ASC,
    "desc":     TokenType.DESC,
    "in":       TokenType.IN,
}

_MASTER_RE = re.compile(
    "|".join(
        f"(?P<G{i}_{re.sub(r'[^A-Za-z0-9]', '_', name)}>{pat})"
        for i, (name, pat) in enumerate(_TOKEN_PATTERNS)
    ),
    re.IGNORECASE,
)


def tokenize(query: str) -> list[Token]:
    """Lex *query* into a list of :class:`Token` objects.

    Args:
        query: Raw MQL query string.

    Returns:
        List of tokens (the final token is always ``EOF``).

    Raises:
        MQLParseError: On unrecognised characters.
    """
    tokens: list[Token] = []
    pos = 0
    n = len(query)

    while pos < n:
        # Skip whitespace
        if query[pos].isspace():
            pos += 1
            continue

        m = _MASTER_RE.match(query, pos)
        if not m:
            raise MQLParseError(query, f"Unexpected character {query[pos]!r}", position=pos)

        raw = m.group(0)
        # Determine which group matched
        for i, (name, _) in enumerate(_TOKEN_PATTERNS):
            group_key = f"G{i}_{re.sub(r'[^A-Za-z0-9]', '_', name)}"
            if m.group(group_key) is not None:
                tok_type = name
                break

        # Post-process
        if tok_type == TokenType.IDENT:
            tok_type = _KEYWORDS.get(raw.lower(), TokenType.IDENT)
        elif tok_type == TokenType.STRING:
            # Strip quotes
            raw = raw[1:-1]

        tokens.append(Token(tok_type, raw, pos))
        pos = m.end()

    tokens.append(Token(TokenType.EOF, "", pos))
    return tokens


# ── Parser ────────────────────────────────────────────────────────────────────

class MQLParser:
    """Recursive-descent MQL parser.

    Converts an MQL string into a :class:`SelectStatement` AST.

    Example::

        parser = MQLParser()
        stmt = parser.parse('SELECT memory FROM / WHERE SIMILAR TO "auth bug" LIMIT 5')
        assert stmt.limit == 5
    """

    def parse(self, query: str) -> SelectStatement:
        """Parse *query* into a :class:`SelectStatement`.

        Args:
            query: MQL string.

        Returns:
            :class:`SelectStatement` AST root.

        Raises:
            MQLParseError: On syntax errors.
        """
        self._query = query
        self._tokens = tokenize(query)
        self._pos = 0
        return self._parse_select()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _peek(self) -> Token:
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        if tok.type != TokenType.EOF:
            self._pos += 1
        return tok

    def _expect(self, *types: str) -> Token:
        tok = self._peek()
        if tok.type not in types:
            raise MQLParseError(
                self._query,
                f"Expected {' or '.join(types)}, got {tok.type!r} ({tok.value!r})",
                position=tok.position,
            )
        return self._advance()

    def _match(self, *types: str) -> bool:
        return self._peek().type in types

    # ── Grammar rules ─────────────────────────────────────────────────────────

    def _parse_select(self) -> SelectStatement:
        self._expect(TokenType.SELECT)
        self._expect(TokenType.MEMORY)
        self._expect(TokenType.FROM)

        path = self._parse_path()
        conditions = None
        limit = None
        order_by = None
        order_dir = "desc"

        if self._match(TokenType.WHERE):
            self._advance()
            conditions = self._parse_condition()

        if self._match(TokenType.ORDER):
            self._advance()
            self._expect(TokenType.BY)
            field_tok = self._expect(TokenType.IDENT, TokenType.DATE, TokenType.CREATED, TokenType.MODIFIED)
            order_by = field_tok.value.lower()
            if self._match(TokenType.ASC):
                self._advance()
                order_dir = "asc"
            elif self._match(TokenType.DESC):
                self._advance()
                order_dir = "desc"

        if self._match(TokenType.LIMIT):
            self._advance()
            n_tok = self._expect(TokenType.INTEGER)
            limit = int(n_tok.value)

        self._expect(TokenType.EOF)
        return SelectStatement(
            path=path,
            conditions=conditions,
            limit=limit,
            order_by=order_by,
            order_dir=order_dir,
        )

    def _parse_path(self) -> str:
        """Parse a memory path (``/``, ``/knowledge``, or a quoted string)."""
        tok = self._peek()
        if tok.type == TokenType.PATH:
            self._advance()
            return tok.value
        if tok.type == TokenType.STRING:
            self._advance()
            return tok.value
        # Bare slash
        if tok.type == TokenType.SLASH:
            self._advance()
            return "/"
        # Bare identifier (treat as path component)
        if tok.type == TokenType.IDENT:
            self._advance()
            return f"/{tok.value}"
        raise MQLParseError(
            self._query,
            f"Expected a path after FROM, got {tok.type!r} ({tok.value!r})",
            position=tok.position,
        )

    def _parse_condition(self) -> Any:
        """Parse a condition expression (handles AND / OR)."""
        left = self._parse_simple_condition()

        while self._match(TokenType.AND, TokenType.OR):
            op_tok = self._advance()
            right = self._parse_simple_condition()
            if op_tok.type == TokenType.AND:
                left = AndCondition(left=left, right=right)
            else:
                left = OrCondition(left=left, right=right)

        return left

    def _parse_simple_condition(self) -> Any:
        tok = self._peek()

        if tok.type == TokenType.SIMILAR:
            return self._parse_similar()
        if tok.type == TokenType.TAG:
            return self._parse_tag()
        if tok.type in (TokenType.DATE, TokenType.CREATED, TokenType.MODIFIED):
            return self._parse_date()
        if tok.type == TokenType.TOPIC:
            return self._parse_topic()
        if tok.type == TokenType.RELATED:
            return self._parse_related()
        if tok.type == TokenType.LPAREN:
            self._advance()
            cond = self._parse_condition()
            self._expect(TokenType.RPAREN)
            return cond

        raise MQLParseError(
            self._query,
            f"Expected a condition keyword (SIMILAR, TAG, DATE, TOPIC, RELATED), got {tok.type!r}",
            position=tok.position,
        )

    def _parse_similar(self) -> SimilarCondition:
        self._expect(TokenType.SIMILAR)
        self._expect(TokenType.TO)
        s = self._expect(TokenType.STRING)
        return SimilarCondition(query_str=s.value)

    def _parse_tag(self) -> TagCondition:
        self._expect(TokenType.TAG)
        op_tok = self._expect(TokenType.EQ, TokenType.NEQ, TokenType.IN)
        op = op_tok.type.lower()

        if op_tok.type == TokenType.IN:
            self._expect(TokenType.LPAREN)
            values: list[str] = []
            while not self._match(TokenType.RPAREN):
                v = self._expect(TokenType.STRING)
                values.append(v.value)
                if self._match(TokenType.COMMA):
                    self._advance()
            self._expect(TokenType.RPAREN)
            return TagCondition(tag="", op="in", values=values)

        val = self._expect(TokenType.STRING)
        return TagCondition(tag=val.value, op=op, values=[val.value])

    def _parse_date(self) -> DateCondition:
        field_tok = self._advance()  # DATE | CREATED | MODIFIED
        op_tok = self._expect(TokenType.GT, TokenType.LT, TokenType.GTE, TokenType.LTE, TokenType.EQ)
        val = self._expect(TokenType.STRING)
        return DateCondition(
            field=field_tok.value.lower(),
            op=op_tok.value,
            value=val.value,
        )

    def _parse_topic(self) -> TopicCondition:
        self._expect(TokenType.TOPIC)
        s = self._expect(TokenType.STRING)
        return TopicCondition(topic_str=s.value)

    def _parse_related(self) -> RelatedToCondition:
        self._expect(TokenType.RELATED)
        self._expect(TokenType.TO)
        s = self._expect(TokenType.STRING, TokenType.PATH)
        depth = 2
        if self._match(TokenType.WITHIN):
            self._advance()
            n_tok = self._expect(TokenType.INTEGER)
            depth = int(n_tok.value)
        return RelatedToCondition(anchor_path=s.value, depth=depth)
