"""Persistence for domain-adapted attack cases (attacker/case_generation.py),
one file per reconstructed environment keyed by its config hash.

Generated once per environment, then reused unchanged forever. That's not
a caching optimisation, it's a correctness requirement: stats/ pairs runs
on case id, a paired comparison must face byte-identical cases in both
arms, and a re-test months later is only comparable to the first if the
suite didn't drift underneath it. Regenerating per execution would quietly
break all three, and the drift would be invisible in the results.

Same discipline as target_system/config.py's config store — content lives
under the user data dir (config/paths.py explains why that isn't
package-relative), written once, read back by hash.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from attacker.attack_case import AttackCase
from attacker.case_generation import GeneratedCase
from config import paths

DEFAULT_GENERATED_CASES_DIR = paths.GENERATED_CASES_DIR

SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class GeneratedCaseSet:
    """What was generated for one environment, with the review state that
    decides whether it may run."""

    config_hash: str
    schema_version: str
    generated_at: str
    model: str
    approved: bool
    entries: tuple[GeneratedCase, ...]

    @property
    def cases(self) -> list[AttackCase]:
        return [e.case for e in self.entries]

    @property
    def coherent_cases(self) -> list[AttackCase]:
        return [e.case for e in self.entries if e.coherent]


def generated_cases_path(config_hash: str, *, cases_dir: Path = DEFAULT_GENERATED_CASES_DIR) -> Path:
    return cases_dir / f"{config_hash}.json"


def save_generated_cases(
    config_hash: str,
    entries: list[GeneratedCase],
    *,
    model: str,
    generated_at: str,
    approved: bool = False,
    cases_dir: Path = DEFAULT_GENERATED_CASES_DIR,
) -> Path:
    """Writes (or overwrites) this environment's generated set.

    Overwrites deliberately, unlike save_config: regenerating is an
    explicit user action here, not an incidental side effect of running a
    test, and the whole point of the store is that nothing regenerates on
    its own. `approved` starts False — a set that hasn't been reviewed
    cannot be picked up by the run path (see load_generated_cases callers)."""
    cases_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "config_hash": config_hash,
        "generated_at": generated_at,
        "model": model,
        "approved": approved,
        "entries": [
            {
                "template_id": e.template_id,
                "template_benign_task": e.template_benign_task,
                "coherent": e.coherent,
                "coherence_reason": e.coherence_reason,
                "case": e.case.model_dump(mode="json"),
            }
            for e in entries
        ],
    }
    path = generated_cases_path(config_hash, cases_dir=cases_dir)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_generated_cases(
    config_hash: str, *, cases_dir: Path = DEFAULT_GENERATED_CASES_DIR
) -> GeneratedCaseSet | None:
    """None when this environment has no generated set yet — callers fall
    back to the hand-authored suite rather than failing."""
    path = generated_cases_path(config_hash, cases_dir=cases_dir)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return GeneratedCaseSet(
        config_hash=data["config_hash"],
        schema_version=data.get("schema_version", SCHEMA_VERSION),
        generated_at=data.get("generated_at", ""),
        model=data.get("model", ""),
        approved=bool(data.get("approved", False)),
        entries=tuple(
            GeneratedCase(
                case=AttackCase.model_validate(entry["case"]),
                template_id=entry["template_id"],
                template_benign_task=entry.get("template_benign_task", ""),
                coherent=bool(entry.get("coherent", True)),
                coherence_reason=entry.get("coherence_reason", ""),
            )
            for entry in data.get("entries", [])
        ),
    )


def approve_generated_cases(config_hash: str, *, cases_dir: Path = DEFAULT_GENERATED_CASES_DIR) -> None:
    """Flips the reviewed flag after a human has actually seen the cases.
    Separate from save so approval is always an explicit act, never a
    default that generation could grant itself."""
    path = generated_cases_path(config_hash, cases_dir=cases_dir)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["approved"] = True
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
