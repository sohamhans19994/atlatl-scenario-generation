import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


def _load_replay_js(path: Path) -> List[Dict[str, Any]]:
    """
    Load a JS replay file of the form:

        replayData = [
        '{"type": "...", ...}',
        ...
        ]

    and return a list of parsed JSON messages.
    """
    text = path.read_text()

    # Extract the bracketed array after 'replayData ='
    marker = "replayData"
    idx = text.find(marker)
    if idx == -1:
        raise ValueError(f"{path} does not look like an ATLATL replay (missing 'replayData').")

    bracket_start = text.find("[", idx)
    bracket_end = text.rfind("]")
    if bracket_start == -1 or bracket_end == -1 or bracket_end <= bracket_start:
        raise ValueError(f"{path} does not contain a well-formed replay array.")

    array_text = text[bracket_start + 1 : bracket_end]  # drop outer brackets

    messages: List[Dict[str, Any]] = []
    for line in array_text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Lines are of the form:  'JSON',  or  'JSON'
        if line.endswith(","):
            line = line[:-1].rstrip()
        if line.startswith("'") and line.endswith("'"):
            line = line[1:-1]
        if not line:
            continue
        messages.append(json.loads(line))

    return messages


def _split_reps(messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Split a flat list of messages into per-rep chunks.

    Each game / repetition starts with a 'parameters' message.
    """
    reps: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []

    for msg in messages:
        if msg.get("type") == "parameters":
            if current:
                reps.append(current)
                current = []
        current.append(msg)

    if current:
        reps.append(current)

    return reps


def _index_units_by_id(units: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Index units by their uniqueId. Falls back to faction + longName if needed.
    """
    result: Dict[str, Dict[str, Any]] = {}
    for u in units:
        uid = u.get("uniqueId")
        if not uid:
            uid = f"{u.get('faction', '')} {u.get('longName', '')}"
        result[uid] = u
    return result


def _compute_rep_metrics(
    rep_messages: List[Dict[str, Any]]
) -> Tuple[Optional[float], float, float, Optional[float]]:
    """
    Compute metrics for a single repetition:

    - episode length (phaseCount from final status)
    - total damage dealt by blue (to red units)
    - total damage taken by blue
    - ratio damage_dealt / damage_taken (None if denominator is 0)
    """
    initial_units: Optional[List[Dict[str, Any]]] = None
    final_obs: Optional[Dict[str, Any]] = None

    for msg in rep_messages:
        if msg.get("type") == "parameters" and initial_units is None:
            params = msg.get("parameters", {})
            initial_units = params.get("units", [])
        elif msg.get("type") == "observation":
            obs = msg.get("observation", {})
            # Keep overwriting; last observation is terminal state
            final_obs = obs

    if not initial_units or not final_obs:
        return None, 0.0, 0.0, None

    final_units = final_obs.get("units", [])
    status = final_obs.get("status", {})
    phase_count = float(status.get("phaseCount")) if "phaseCount" in status else None

    init_by_id = _index_units_by_id(initial_units)
    final_by_id = _index_units_by_id(final_units)

    damage_dealt_blue = 0.0
    damage_taken_blue = 0.0

    for uid, u0 in init_by_id.items():
        faction = u0.get("faction")
        init_strength = float(u0.get("currentStrength", 0.0))
        u1 = final_by_id.get(uid)
        final_strength = float(u1.get("currentStrength", 0.0)) if u1 is not None else 0.0
        dmg = max(0.0, init_strength - max(final_strength, 0.0))

        if faction == "red":
            damage_dealt_blue += dmg
        elif faction == "blue":
            damage_taken_blue += dmg

    ratio: Optional[float]
    if damage_taken_blue > 0:
        ratio = damage_dealt_blue / damage_taken_blue
    else:
        ratio = None

    return phase_count, damage_dealt_blue, damage_taken_blue, ratio


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute basic metrics (episode length, total damage dealt / taken, "
            "and their ratio for blue) from ATLATL replay files."
        )
    )
    parser.add_argument(
        "--blue_replay",
        type=Path,
        default=Path("blue_replay.js"),
        help="Path to blue replay JS file (default: blue_replay.js).",
    )
    parser.add_argument(
        "--red_replay",
        type=Path,
        default=Path("red_replay.js"),
        help=(
            "Path to red replay JS file (not strictly needed for these metrics, "
            "but kept for symmetry)."
        ),
    )

    args = parser.parse_args()

    blue_msgs = _load_replay_js(args.blue_replay)
    # Red replay is not used for current metrics, but we validate it parses OK.
    if args.red_replay.exists():
        _ = _load_replay_js(args.red_replay)

    reps = _split_reps(blue_msgs)
    if not reps:
        print("No repetitions found in replay.")
        return

    lengths: List[float] = []
    dealt_list: List[float] = []
    taken_list: List[float] = []
    ratios: List[float] = []

    print(f"Found {len(reps)} repetition(s) in replay.")

    for i, rep in enumerate(reps, start=1):
        phase_count, dealt, taken, ratio = _compute_rep_metrics(rep)
        lengths.append(phase_count if phase_count is not None else 0.0)
        dealt_list.append(dealt)
        taken_list.append(taken)
        if ratio is not None:
            ratios.append(ratio)

        print(
            f"Rep {i}: "
            f"phases={phase_count}, "
            f"blue_damage_dealt={dealt:.2f}, "
            f"blue_damage_taken={taken:.2f}, "
            f"ratio_dealt_to_taken={ratio if ratio is not None else 'N/A'}"
        )

    n = len(reps)
    avg_len = sum(lengths) / n
    avg_dealt = sum(dealt_list) / n
    avg_taken = sum(taken_list) / n
    avg_ratio = sum(ratios) / len(ratios) if ratios else None

    print("\n=== Averages over all repetitions ===")
    print(f"Average phases per rep: {avg_len:.2f}")
    print(f"Average blue damage dealt: {avg_dealt:.2f}")
    print(f"Average blue damage taken: {avg_taken:.2f}")
    print(
        "Average blue damage ratio (dealt / taken): "
        f"{avg_ratio:.4f}" if avg_ratio is not None else "Average ratio: N/A"
    )


if __name__ == "__main__":
    main()




