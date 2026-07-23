from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ATTRIBUTE_QUERY_FUNCS = {"query_color", "query_material", "query_size", "query_shape"}
COUNTING_FUNCS = {"count"}
EXISTENCE_FUNCS = {"exist"}
INTEGER_COMPARE_FUNCS = {"equal_integer", "less_than", "greater_than"}
ATTRIBUTE_COMPARE_FUNCS = {"equal_color", "equal_material", "equal_size", "equal_shape"}
RELATIONAL_FUNCS = {
    "relate",
    "same_size",
    "same_color",
    "same_material",
    "same_shape",
}

CATEGORY_FROM_TERMINAL = {
    **{name: "attribute_query" for name in ATTRIBUTE_QUERY_FUNCS},
    **{name: "counting" for name in COUNTING_FUNCS},
    **{name: "existence" for name in EXISTENCE_FUNCS},
    **{name: "integer_comparison" for name in INTEGER_COMPARE_FUNCS},
    **{name: "attribute_comparison" for name in ATTRIBUTE_COMPARE_FUNCS},
}

PROGRAM_LENGTH_BINS = (
    ("1-5", 1, 5),
    ("6-10", 6, 10),
    ("11-15", 11, 15),
    ("16+", 16, 10_000),
)


@dataclass(frozen=True)
class ProgramStats:
    n_operations: int
    dependency_depth: int
    operation_types: tuple[str, ...]
    n_relational_ops: int
    terminal_function: str
    reasoning_category: str
    program_length_bin: str


def _program_nodes(program: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return list(program or [])


def dependency_chain_depth(program: list[dict[str, Any]] | None) -> int:
    nodes = _program_nodes(program)
    if not nodes:
        return 0
    depths = [0] * len(nodes)
    for index, node in enumerate(nodes):
        inputs = node.get("inputs") or []
        if not inputs:
            depths[index] = 1
        else:
            depths[index] = 1 + max(depths[int(parent)] for parent in inputs)
    return int(max(depths))


def program_length_bin(n_operations: int) -> str:
    for label, low, high in PROGRAM_LENGTH_BINS:
        if low <= n_operations <= high:
            return label
    return "16+"


def analyze_program(program: list[dict[str, Any]] | None) -> ProgramStats:
    nodes = _program_nodes(program)
    op_types = tuple(str(node.get("function", "")) for node in nodes)
    terminal = op_types[-1] if op_types else ""
    category = CATEGORY_FROM_TERMINAL.get(terminal, "other")
    n_rel = sum(1 for name in op_types if name in RELATIONAL_FUNCS)
    n_ops = len(nodes)
    return ProgramStats(
        n_operations=n_ops,
        dependency_depth=dependency_chain_depth(nodes),
        operation_types=op_types,
        n_relational_ops=n_rel,
        terminal_function=terminal,
        reasoning_category=category,
        program_length_bin=program_length_bin(n_ops),
    )


def question_mentions_shape(question: str, shape: str) -> bool:
    return shape.lower() in question.lower().split() or shape.lower() in question.lower()
