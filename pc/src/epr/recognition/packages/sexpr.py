"""A minimal s-expression reader for KiCad files.

KiCad footprints are s-expressions, not a format any dependency in this project reads. The
subset needed here is small: nested lists, bare atoms and quoted strings. Numbers are left as
text and converted by the caller, because a footprint mixes coordinates with tokens like
``smd`` and layer names.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

Node = list["str | Node"]

TOKEN = re.compile(r'\s*(?:([()])|"((?:[^"\\]|\\.)*)"|([^\s()"]+))')
ESCAPE = re.compile(r"\\(.)")


def parse(text: str) -> Node:
    """Parse a document into nested lists. Raises on unbalanced parentheses."""
    root: Node = []
    stack: list[Node] = [root]

    for match in TOKEN.finditer(text):
        paren, quoted, atom = match.groups()
        if paren == "(":
            node: Node = []
            stack[-1].append(node)
            stack.append(node)
        elif paren == ")":
            if len(stack) == 1:
                raise ValueError("unbalanced closing parenthesis")
            stack.pop()
        elif quoted is not None:
            stack[-1].append(ESCAPE.sub(r"\1", quoted))
        else:
            stack[-1].append(atom)

    if len(stack) != 1:
        raise ValueError("unbalanced opening parenthesis")
    return root


def tagged(node: Node, tag: str) -> Iterator[Node]:
    """Direct children that are lists beginning with ``tag``."""
    for child in node:
        if isinstance(child, list) and child and child[0] == tag:
            yield child


def walk(node: Node) -> Iterator[Node]:
    """Every list in the tree, including the node itself."""
    yield node
    for child in node:
        if isinstance(child, list):
            yield from walk(child)


def first(node: Node, tag: str) -> Node | None:
    return next(tagged(node, tag), None)


def atom(node: Node, tag: str, index: int = 1) -> str | None:
    """The ``index``-th atom of the first child named ``tag``."""
    child = first(node, tag)
    if child is None or len(child) <= index:
        return None
    value = child[index]
    return value if isinstance(value, str) else None


def floats(node: Node | None, count: int = 2) -> tuple[float, ...] | None:
    """Read ``count`` numbers from a node such as ``(start -1 -0.625)``."""
    if node is None or len(node) <= count:
        return None
    try:
        return tuple(float(value) for value in node[1 : count + 1])  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
