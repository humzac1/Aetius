"""The single versioned config object that drives every knob of the target
system. Every run's trajectory log references a config only by its content
hash (see compute_config_hash) — the resolved config itself is persisted
once to target_system/configs/<hash>.json so it can be looked up later.

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
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

CONFIG_SCHEMA_VERSION = "1.0"

DEFAULT_CONFIGS_DIR = Path(__file__).parent / "configs"


class ModelConfig(BaseModel):
    provider: Literal["anthropic", "mock"]
    model_name: str
    temperature: float = 0.0
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

    def supervisor(self) -> AgentSpec:
        for a in self.agents:
            if a.role == "supervisor":
                return a
        raise ValueError("config has no agent with role='supervisor'")

    def members(self) -> list[AgentSpec]:
        return [a for a in self.agents if a.role != "supervisor"]


def _canonical_json(config: SystemConfig) -> str:
    data = config.model_dump(mode="json", exclude={"label"})
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
