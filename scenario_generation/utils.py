"""Map conversion, validation, and LLM-call utilities."""

from __future__ import annotations

import json
import re
from typing import Any, Callable


class HexMapValidator:
    """Validate the compact, staggered hex-map text format."""

    CELL_RE = re.compile(r"\A[curmws](?:[1-9]\d*[itmad][br])?\Z")

    @staticmethod
    def _parse_even_line(line: str) -> tuple[list[str], str | None]:
        if line.startswith("\t"):
            return [], "Even line has leading tabs; expected a cell."
        match = re.match(r"^(.*?)(\t{0,3})$", line)
        assert match is not None
        parts = match.group(1).split("\t\t\t\t")
        if any("\t" in part for part in parts):
            return [], "Even line uses tabs outside four-tab separators."
        return parts, None

    @staticmethod
    def _parse_odd_line(line: str) -> tuple[list[str], str | None]:
        if not line.startswith("\t\t"):
            return [], "Odd line must start with exactly two tabs."
        if line.startswith("\t\t\t"):
            return [], "Odd line starts with more than two tabs."
        match = re.match(r"^(.*?)(\t{0,3})$", line[2:])
        assert match is not None
        parts = match.group(1).split("\t\t\t\t")
        if any("\t" in part for part in parts):
            return [], "Odd line uses tabs outside four-tab separators."
        return parts, None

    @classmethod
    def _validate_cell(cls, token: str) -> str | None:
        if not token:
            return "Empty cell."
        if " " in token:
            return "Cell contains spaces."
        if not cls.CELL_RE.fullmatch(token):
            return (
                "Invalid cell syntax; expected terrain [c,u,r,m,w,s] and an "
                "optional <strength><unit:[i,t,m,a,d]><team:[b,r]> triplet."
            )
        return None

    @classmethod
    def validate_map_text(cls, text: str) -> dict[str, Any]:
        lines = text.splitlines()
        errors: list[str] = []
        if len(lines) % 2:
            errors.append(
                f"Text has {len(lines)} lines; each map row requires two lines."
            )

        rows: list[dict[str, Any]] = []
        expected_cols: int | None = None
        for line_index in range(0, len(lines), 2):
            row_index = line_index // 2
            even_line = lines[line_index]
            odd_line = lines[line_index + 1] if line_index + 1 < len(lines) else ""
            even_cells, even_error = cls._parse_even_line(even_line)
            odd_cells, odd_error = cls._parse_odd_line(odd_line)

            if even_error:
                errors.append(f"Row {row_index}: {even_error}")
            if odd_error:
                errors.append(f"Row {row_index}: {odd_error}")

            for group_name, cells in (
                ("even", even_cells),
                ("odd", odd_cells),
            ):
                for cell_index, cell in enumerate(cells):
                    cell_error = cls._validate_cell(cell)
                    if cell_error:
                        errors.append(
                            f"Row {row_index} {group_name} cell {cell_index}: "
                            f"{cell_error} Got {cell!r}."
                        )

            total_cols = len(even_cells) + len(odd_cells)
            if expected_cols is None:
                expected_cols = total_cols
            elif total_cols != expected_cols:
                errors.append(
                    f"Row {row_index} has {total_cols} columns; "
                    f"expected {expected_cols}."
                )
            if abs(len(even_cells) - len(odd_cells)) > 1:
                errors.append(
                    f"Row {row_index} has invalid staggering "
                    f"({len(even_cells)} even, {len(odd_cells)} odd)."
                )

            rows.append(
                {
                    "row_index": row_index,
                    "even_cells": even_cells,
                    "odd_cells": odd_cells,
                    "total_cols": total_cols,
                }
            )

        return {
            "ok": not errors,
            "errors": errors,
            "num_rows": len(rows),
            "expected_cols": expected_cols,
            "rows": rows,
        }


def observation_to_str(
    parameters: dict[str, Any],
    observation: dict[str, Any],
    *,
    include_on_move: bool = True,
) -> str:
    """Render a portable AtlAtl observation without importing AtlAtl internals."""
    terrain_codes = {
        "clear": "c",
        "urban": "u",
        "rough": "r",
        "marsh": "m",
        "water": "w",
        "coast": "s",
    }
    unit_codes = {
        "infantry": "i",
        "armor": "t",
        "mechinf": "m",
        "artillery": "a",
        "ddg": "d",
    }

    hexes = parameters["map"]["hexes"]
    hex_by_coordinate = {
        (int(item["x_offset"]), int(item["y_offset"])): item for item in hexes
    }
    unit_by_hex = {
        unit["hex"]: unit
        for unit in observation.get("units", [])
        if unit.get("hex") and not unit.get("ineffective", False)
    }
    city_owners = observation.get("status", {}).get("cityOwner", {})
    width = max(x for x, _ in hex_by_coordinate) + 1
    height = max(y for _, y in hex_by_coordinate) + 1

    def render_hex(x: int, y: int) -> str:
        item = hex_by_coordinate[(x, y)]
        hex_id = f"hex-{x}-{y}"
        result = terrain_codes[item["terrain"]]
        if result == "u":
            owner = city_owners.get(hex_id)
            if owner in {"blue", "red"}:
                result += owner[0]

        unit = unit_by_hex.get(hex_id)
        if unit:
            result += str(int(unit["currentStrength"]))
            result += unit_codes[unit["type"]]
            result += unit["faction"][0]
            if include_on_move and unit.get("canMove", False):
                result += "o"
        return result

    rendered_lines: list[str] = []
    for y in range(height):
        even_cells = [render_hex(x, y) for x in range(0, width, 2)]
        odd_cells = [render_hex(x, y) for x in range(1, width, 2)]
        rendered_lines.append("\t\t\t\t".join(even_cells))
        odd_line = "\t\t" + "\t\t\t\t".join(odd_cells)
        if width % 2:
            odd_line += "\t\t"
        rendered_lines.append(odd_line)
    return "\n".join(rendered_lines) + "\n"


def str_to_scenario(scenario_str: str) -> dict[str, Any]:
    """Convert compact map text into AtlAtl's portable scenario structure."""
    terrain_types = {
        "c": "clear",
        "u": "urban",
        "r": "rough",
        "m": "marsh",
        "w": "water",
        "s": "coast",
    }
    unit_types = {
        "i": "infantry",
        "t": "armor",
        "m": "mechinf",
        "a": "artillery",
        "d": "ddg",
    }
    next_unit_number = {"blue": 1, "red": 1}

    def make_hex(
        terrain_char: str, x_offset: int, y_offset: int
    ) -> tuple[dict[str, Any], set[str]]:
        x_grid = x_offset * 3 + 2
        y_grid = y_offset * 2 + 2 + (x_offset % 2)
        offsets = [(-1, -1), (1, -1), (2, 0), (1, 1), (-1, 1), (-2, 0)]
        vertices = [(x_grid + dx, y_grid + dy) for dx, dy in offsets]
        edge_ids = {
            f"edge-{a[0]}-{a[1]}-{b[0]}-{b[1]}"
            for a, b in zip(vertices, vertices[1:] + vertices[:1])
        }
        return (
            {
                "x_offset": x_offset,
                "y_offset": y_offset,
                "terrain": terrain_types[terrain_char],
                "x_grid": x_grid,
                "y_grid": y_grid,
                "setup": None,
                "edges": list(edge_ids),
            },
            edge_ids,
        )

    def make_unit(unit_text: str, x_offset: int, y_offset: int) -> dict[str, Any]:
        match = re.fullmatch(r"(\d+)([itmad])([br])", unit_text)
        if not match:
            raise ValueError(f"Invalid unit token: {unit_text!r}")
        strength = int(match.group(1))
        faction = {"b": "blue", "r": "red"}[match.group(3)]
        number = next_unit_number[faction]
        next_unit_number[faction] += 1
        if faction == "blue":
            long_name = f"{'A' * number}/1/1"
        else:
            long_name = f"{'B' * number}/1/{number}"
        unique_id = f"{faction} {long_name}"
        return {
            "type": unit_types[match.group(2)],
            "echelon": "regiment",
            "name": 1,
            "faction": faction,
            "longName": long_name,
            "uniqueId": unique_id,
            "currentStrength": strength,
            "fullStrength": strength,
            "homeOrgId": None,
            "taskOrgId": None,
            "hex": f"hex-{x_offset}-{y_offset}",
            "canMove": False,
            "ineffective": False,
        }

    scenario: dict[str, Any] = {
        "map": {"hexes": [], "edges": [], "paths": [], "fogOfWar": False},
        "units": [],
        "score": {"maxPhases": 50, "lossPenalty": -1, "cityScore": 24},
    }
    seen_edges: set[str] = set()

    for line_index, row in enumerate(scenario_str.splitlines()):
        x_offset = line_index % 2
        y_offset = line_index // 2
        cells = [part for part in re.split(r"\t{4,}", row.strip()) if part]
        for cell in cells:
            hex_item, edge_ids = make_hex(cell[0], x_offset, y_offset)
            scenario["map"]["hexes"].append(hex_item)
            for edge_id in edge_ids - seen_edges:
                _, xa, ya, xb, yb = edge_id.split("-")
                scenario["map"]["edges"].append(
                    {
                        "id": edge_id,
                        "xa_grid": xa,
                        "ya_grid": ya,
                        "xb_grid": xb,
                        "yb_grid": yb,
                        "type": "normal",
                    }
                )
                seen_edges.add(edge_id)
            if len(cell) > 1:
                scenario["units"].append(make_unit(cell[1:], x_offset, y_offset))
            x_offset += 2
    return scenario


def safe_llm_complete(
    prompt_text: str,
    llm: Any | None = None,
    *,
    raise_on_error: bool = False,
    max_tries: int = 3,
    validator: Callable[..., dict[str, Any]] | None = None,
    logging: Any | None = None,
    **validator_kwargs: Any,
) -> dict[str, Any]:
    """Call an LLM with bounded retries and optional response validation."""
    if llm is None:
        raise ValueError("An explicitly configured LLM client is required")

    errors: list[Any] = []
    for _ in range(max_tries):
        try:
            response = llm.complete(prompt_text)
            raw_text = getattr(response, "text", None) or str(response)
            validation_result = None
            if validator is not None:
                validation_result = validator(raw_text, **validator_kwargs)
                if not validation_result["ok"]:
                    errors.extend(validation_result["errors"])
                    if logging is not None:
                        logging.simple_log(
                            f"Validation error: {validation_result['errors']}"
                        )
                    continue
            return {
                "ok": True,
                "response_str": raw_text,
                "validation_result": validation_result,
                "errors": errors,
            }
        except Exception as exc:  # LLM providers expose multiple exception types.
            errors.append({"type": type(exc).__name__, "message": str(exc)})

    result = {
        "ok": False,
        "response_str": None,
        "validation_result": None,
        "errors": errors,
    }
    if raise_on_error:
        raise RuntimeError(errors[-1] if errors else "LLM completion failed")
    return result


# Compatibility aliases for existing notebooks that used the original names.
strToScenario = str_to_scenario
observationToStr = observation_to_str
