"""The single versioned config object that drives every knob of the target
system. Every run's trajectory log references a config only by its content
hash (see compute_config_hash) — the resolved config itself is persisted
once to <user data dir>/configs/<hash>.json so it can be looked up later
(config/paths.py resolves that, and explains why it isn't package-relative).

Design choices that matter downstream:
  - AgentSpec.system_prompt is the fully resolved prompt text, not a
    template. prompts.py composes named building blocks (base prompt +
    optional defensive clause) into that final string *before* the config
    is constructed, so the hash captures exactly what the model saw and two
    configs can never collide on hash while having actually run different
    text.
  - `agents` is an ordered list, not fixed fields, so "add a fourth agent"
    (the Part 4 preset-5 experiment) is just appending an AgentSpec with a
    new role — no schema migration.
  - `label` is intentionally excluded from the hash: it's a human note, not
    part of the experimental condition. Two configs with different labels
    but identical resolved content are the same condition and must collide.
  - `provenance` (target_system/provenance.py) is excluded from the hash for
    the same reason: it's set by ingestion/reconstruct.py to describe where
    a reconstructed config came from (trace count, project, warnings), not
    part of what the config resolves to. None for every hand-authored
    (toy system) config.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from config import paths
from target_system.provenance import ReconstructionProvenance

CONFIG_SCHEMA_VERSION = "1.0"

DEFAULT_CONFIGS_DIR = paths.CONFIGS_DIR


class ModelConfig(BaseModel):
    provider: Literal["anthropic", "mock"]
    model_name: str
    # None means "not pinned" -- omitted from the API call entirely, so the
    # provider's own default applies. Reconstruction never observes a real
    # temperature from trace data (Braintrust/Langfuse don't record sampling
    # params), so a reconstructed config leaves this None rather than
    # fabricating 0.0; some real models (e.g. claude-opus-4-8, confirmed via
    # a real Braintrust-reconstructed end-to-end run) reject an explicit
    # temperature outright with a 400 "temperature is deprecated for this
    # model" error, so fabricating a value here isn't just inaccurate, it's
    # actively unsafe to send. The toy system (target_system/factory.py)
    # still pins 0.0 explicitly for its own determinism.
    temperature: float | None = 0.0
    max_tokens: int = 2048
    # Passed through to the API where supported (Anthropic does not expose a
    # sampling seed today, but this stays wired through the config/adapter
    # so common-random-numbers pairing degrades gracefully rather than
    # silently doing nothing when the API adds support).
    seed: int | None = None


class AgentSpec(BaseModel):
    role: str
    name: str
    system_prompt: str
    # "observed": system_prompt is real text seen in the source (hand-authored
    # configs, or a future reconstruction source that actually captures it).
    # "unavailable": system_prompt is a disclosed placeholder, never a
    # fabrication — see ingestion/reconstruct.py's investigation notes on why
    # Langfuse-sourced traces don't carry this text. Downstream (Part 6) must
    # surface "unavailable" inline on any verdict for this config.
    system_prompt_source: Literal["observed", "unavailable"] = "observed"
    tools: list[str] = Field(default_factory=list)
    model_override: ModelConfig | None = None


class SecurityConfig(BaseModel):
    email_allowlist: list[str] = Field(default_factory=list)
    # Explicit, resolved filenames — not "N% at seed X" — so the config hash
    # captures the exact experimental condition and two runs of "the same"
    # config can never silently poison a different subset of the corpus.
    poisoned_corpus_files: list[str] = Field(default_factory=list)
    # Same idea as poisoned_corpus_files but for structured tool results
    # (e.g. a customer record's notes field) rather than document bodies —
    # the other injection vector attacker/cases.py's tool_result_poisoning
    # family uses. Holds the keys (e.g. customer IDs) whose results are
    # allowed to carry an attack case's injected content this run.
    poisoned_tool_results: list[str] = Field(default_factory=list)
    # If True, orchestration.py installs a real pre-execution guard on
    # send_email that blocks (rather than just logs) an off-allowlist call.
    # Lets an experiment isolate "the model stopped trying" (a prompt/model
    # effect) from "the system stopped letting it succeed" (an enforcement
    # effect) — see ToolCallEvent.status and policy.py's *_attempted split.
    enforce_allowlist: bool = False


class SystemConfig(BaseModel):
    schema_version: str = CONFIG_SCHEMA_VERSION
    label: str
    model: ModelConfig
    agents: list[AgentSpec]
    security: SecurityConfig
    corpus_dir: str = "target_system/corpus"
    defensive_instruction: bool = False
    # None for every hand-authored (toy system) config. Set only by
    # ingestion/reconstruct.py for a config rebuilt from real traces —
    # excluded from the hash below for the same reason label is: it
    # describes provenance, not the experimental condition itself.
    provenance: ReconstructionProvenance | None = None

    def supervisor(self) -> AgentSpec:
        for a in self.agents:
            if a.role == "supervisor":
                return a
        raise ValueError("config has no agent with role='supervisor'")

    def members(self) -> list[AgentSpec]:
        return [a for a in self.agents if a.role != "supervisor"]


def _canonical_json(config: SystemConfig) -> str:
    data = config.model_dump(mode="json", exclude={"label", "provenance"})
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def compute_config_hash(config: SystemConfig) -> str:
    digest = hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()
    return f"cfg_{digest[:12]}"


def save_config(config: SystemConfig, configs_dir: Path = DEFAULT_CONFIGS_DIR) -> str:
    """Persist the resolved config, keyed by its content hash. Idempotent:
    writing the same config twice is a no-op after the first write."""
    config_hash = compute_config_hash(config)
    configs_dir.mkdir(parents=True, exist_ok=True)
    path = configs_dir / f"{config_hash}.json"
    if not path.exists():
        payload = {"config_hash": config_hash, **config.model_dump(mode="json")}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return config_hash


def load_config(config_hash: str, configs_dir: Path = DEFAULT_CONFIGS_DIR) -> SystemConfig:
    path = configs_dir / f"{config_hash}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.pop("config_hash", None)
    return SystemConfig.model_validate(data)


def list_config_hashes(configs_dir: Path = DEFAULT_CONFIGS_DIR) -> list[str]:
    """Every config_hash saved under configs_dir, newest first — the
    enumeration save_config/load_config never needed on their own (each
    caller already knows the hash it wants), but tui/'s "manage configs"
    screen does. Sorted by file mtime rather than name so a freshly-run
    experiment's configs surface first without the caller needing to
    inspect each file."""
    if not configs_dir.exists():
        return []
    paths = sorted(configs_dir.glob("cfg_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.stem for p in paths]
