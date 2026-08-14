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
  - `provenance` (target_system/provenance.py) is *mostly* excluded from the
    hash: it's set by ingestion/reconstruct.py to describe where a
    reconstructed config came from, not part of what the config resolves to.
    None for every hand-authored (toy system) config. The exception is the
    part of it that genuinely is the environment's content — see
    _provenance_identity below for why excluding it wholesale was a bug.
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


# The fields of ReconstructionProvenance that are the reconstructed
# environment's *content* rather than a note about where it came from.
# tool_profiles is the substantive one: it holds the observed argument
# profiles, response key sets and example (arguments, response) pairs that
# the replay executor actually answers tool calls from, so two pulls with
# different tool_profiles are different environments no matter how alike
# the rest of the config looks. trace_count rides along as the cheap,
# human-legible summary of how much evidence sits behind them.
#
# Everything else stays out, deliberately:
#   - extraction_date is wall-clock time, not data. Folding it in would give
#     every re-pull a fresh identity and destroy dedupe outright.
#   - the avg_* cost/token figures are floats derived from the same traces
#     tool_profiles already covers — redundant discrimination in exchange
#     for a float-repr stability risk across platforms.
#   - warnings and other_groups_found describe the batch around this group,
#     not this environment.
_PROVENANCE_IDENTITY_FIELDS = {"project_id", "source_agent_name", "trace_count", "tool_profiles"}


def _provenance_identity(provenance: ReconstructionProvenance) -> str:
    """Digest of the trace-derived content of a reconstructed config.

    Exists because excluding provenance wholesale from the hash was a
    silent data-loss bug, confirmed in real user data: everything a
    reconstruction learns from its traces lands in provenance, so for a
    Langfuse-sourced config — where system_prompt is the fixed
    "unavailable" placeholder — the hashed content was only the model
    name, the agent name/role, the sorted tool-name list and default
    security settings. Re-pulling the same agent with 100 more traces
    changed none of that, so the second reconstruction hashed to the same
    cfg_* id, save_config's `if not path.exists()` skipped the write, and
    the richer pull was discarded while the UI reported it saved.

    Deterministic across identical pulls: the cached trace ids are loaded
    in sorted order (langfuse_client.load_cached_trace_ids), so the
    capped sample_values/example_calls are drawn in a stable order, and
    sort_keys makes the dict ordering irrelevant. Two pulls that saw the
    same trace data therefore still dedupe to one file."""
    material = provenance.model_dump(mode="json", include=_PROVENANCE_IDENTITY_FIELDS)
    return hashlib.sha256(json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _canonical_json(config: SystemConfig) -> str:
    data = config.model_dump(mode="json", exclude={"label", "provenance"})
    if config.provenance is not None:
        # Added only for reconstructed configs, so every hand-authored (toy
        # system) config keeps the exact hash it had before this key
        # existed — old cfg_* ids in existing run records stay resolvable.
        # The dunder name can't collide with a SystemConfig field: pydantic
        # treats leading-underscore names as private attrs, not fields.
        data["__provenance__"] = _provenance_identity(config.provenance)
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def compute_config_hash(config: SystemConfig) -> str:
    digest = hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()
    return f"cfg_{digest[:12]}"


class ConfigHashCollisionError(RuntimeError):
    """Two configs with different hashed content landed on the same cfg_*
    id — a real 48-bit digest collision (or a corrupt/unreadable file at
    that path). Raised rather than resolved because both outcomes
    available here are wrong: overwriting discards a config earlier runs
    were scored against, and skipping the write is the exact silent
    no-op this module was fixed to stop doing."""


def save_config(config: SystemConfig, configs_dir: Path = DEFAULT_CONFIGS_DIR) -> str:
    """Persist the resolved config, keyed by its content hash. Idempotent:
    writing the same config twice is a no-op after the first write.

    "The same config" means the same *hashed* content, which is what makes
    the no-op safe to take: an existing file is left alone only after its
    canonical form is confirmed byte-identical to what this call would
    write. Fields outside the hash (label, extraction_date, the cost
    averages) may differ between the two and the first write still wins —
    that's the same "different label, same condition" collapse the hash
    has always intended. Anything else raises rather than silently
    keeping the stale file."""
    config_hash = compute_config_hash(config)
    configs_dir.mkdir(parents=True, exist_ok=True)
    path = configs_dir / f"{config_hash}.json"
    if path.exists():
        try:
            existing = load_config(config_hash, configs_dir=configs_dir)
        except Exception as exc:  # unparseable, or written under an older schema
            raise ConfigHashCollisionError(f"{path} exists but could not be read back for comparison: {exc}") from exc
        if _canonical_json(existing) != _canonical_json(config):
            raise ConfigHashCollisionError(
                f"{path} already holds a config with different hashed content under the same id {config_hash!r}"
            )
        return config_hash
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
