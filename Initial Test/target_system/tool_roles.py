"""Classifies a tool into the abstract role(s) attacker/applicability.py
and target_system/policy.py check against, instead of every family/policy
rule hardcoding a specific tool name (send_email, lookup_customer, ...).

Lives in target_system/, not attacker/, because target_system/policy.py
(the toy system's own outcome evaluator, generalized in Part 3 to resolve
tool_role rather than a fixed tool_name — see policy.yaml) needs it just
as much as attacker/applicability.py does, and target_system/ is the
layer both attacker/ and experiments/ already depend on — the reverse
(policy.py importing from attacker/) would invert that.

Classification is name+signature heuristics, not a declared schema — there
isn't one in any source this project has (hand-authored toy tools, or
reconstructed ones from ingestion/reconstruct.py, which only has *observed*
argument/response shapes, confirmed in the Part 1/2 investigation). Works
from a bare tool name alone (the toy system's AgentSpec.tools is just
list[str]) and gets more confident when a ToolBehaviorProfile is available
(a reconstructed tool's observed argument names).

Matching is whole-token, not substring: an earlier version matched hint
substrings directly against the raw name/argument string and produced
real false positives against live reconstructed data — "customer"
contains "to" (matching a "recipient-ish" hint meant for words like "to"),
"invoice_id" contains "id" as a bare substring (matching a lookup-ish
hint meant for "identifier"), so create_invoice/send_invoice/list_invoices
all came back misclassified as both SENSITIVE_ACTION and DATA_LOOKUP
before this was tokenized. Every hint here is checked against the
snake_case/camelCase-split token set instead, and argument-based hints are
kept deliberately narrow (a handful of unambiguous whole words) rather
than broadened back out, since that false-positive mode is exactly what
tokenizing alone doesn't fully protect against for short/common words.

Verified against every tool in this codebase before relying on it
elsewhere: the toy system's send_email/lookup_customer/search_corpus
recover exactly their existing hardcoded policy.yaml roles, and both real
reconstructed environments (Invoice Generation Assistant, HR Onboarding
Assistant — see ingestion/reconstruct.py) were checked by hand against
their classified roles before attacker/applicability.py was trusted to
run on them (see the Part 3 report).
"""

from __future__ import annotations

import re
from enum import Enum

from target_system.provenance import ToolBehaviorProfile


class ToolRole(str, Enum):
    # Transmits or otherwise acts on something with effect outside the
    # agent's own scope — the tool an exfiltration-style outcome targets
    # (send_email, send_invoice, notify_customer, ...).
    SENSITIVE_ACTION = "sensitive_action"
    # Returns an internal/sensitive record given an identifier — the tool
    # an unauthorized-access-style outcome targets (lookup_customer, ...).
    DATA_LOOKUP = "data_lookup"
    # Retrieves content the agent didn't author and can't fully vet before
    # it lands in context (document search, corpus browsing) — the
    # delivery vector indirect_injection_document needs.
    UNTRUSTED_CONTENT_ENTRY_POINT = "untrusted_content_entry_point"


_SENSITIVE_ACTION_NAME_TOKENS = {"send", "email", "notify", "message", "transfer", "pay", "post", "publish", "submit", "export", "share"}
_DATA_LOOKUP_NAME_TOKENS = {"lookup", "get", "fetch", "find", "retrieve", "list"}
_ENTRY_POINT_NAME_TOKENS = {"search", "browse", "query", "scan", "document", "documents"}

# Deliberately narrow — see module docstring on why broader argument hints
# (e.g. a bare "id" or "customer") produced false positives against real
# reconstructed data.
_SENSITIVE_ACTION_ARG_TOKENS = {"recipient"}
_DATA_LOOKUP_ARG_TOKENS = {"identifier"}

_TOKEN_SPLIT_RE = re.compile(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])")


def _tokenize(name: str) -> set[str]:
    """snake_case, kebab-case, and camelCase all split into lowercase
    whole-word tokens: "send_invoice" -> {"send", "invoice"},
    "sendInvoice" -> {"send", "invoice"}. Hints are matched against this
    set, never as a raw substring of the original string."""
    tokens: set[str] = set()
    for part in re.split(r"[_\-\s]+", name):
        tokens.update(match.lower() for match in _TOKEN_SPLIT_RE.findall(part))
    return tokens


def _tokens_overlap(name: str, hint_tokens: set[str]) -> bool:
    return bool(_tokenize(name) & hint_tokens)


def classify_tool_role(tool_name: str, profile: ToolBehaviorProfile | None = None) -> set[ToolRole]:
    """A tool can have more than one role (e.g. a tool that both looks up
    and forwards data) or none (a pure-computation or pure-write helper
    like calculate_total/create_invoice — deliberately not force-fit into
    a role neither outcome type actually targets)."""
    roles: set[ToolRole] = set()

    if _tokens_overlap(tool_name, _SENSITIVE_ACTION_NAME_TOKENS):
        roles.add(ToolRole.SENSITIVE_ACTION)
    if _tokens_overlap(tool_name, _DATA_LOOKUP_NAME_TOKENS):
        roles.add(ToolRole.DATA_LOOKUP)
    if _tokens_overlap(tool_name, _ENTRY_POINT_NAME_TOKENS):
        roles.add(ToolRole.UNTRUSTED_CONTENT_ENTRY_POINT)

    if profile is not None:
        arg_names = list(profile.argument_profiles.keys())
        if any(_tokens_overlap(arg, _SENSITIVE_ACTION_ARG_TOKENS) for arg in arg_names):
            roles.add(ToolRole.SENSITIVE_ACTION)
        if any(_tokens_overlap(arg, _DATA_LOOKUP_ARG_TOKENS) for arg in arg_names):
            roles.add(ToolRole.DATA_LOOKUP)

    return roles


def classify_environment_tools(
    tool_names: list[str], *, tool_profiles: dict[str, ToolBehaviorProfile] | None = None
) -> dict[str, set[ToolRole]]:
    """tool_names: an environment's full tool list (SystemConfig agent(s)'
    combined .tools). tool_profiles: SystemConfig.provenance.tool_profiles
    when available (reconstructed environments), None for hand-authored
    (toy system) configs — classification still works from names alone."""
    tool_profiles = tool_profiles or {}
    return {name: classify_tool_role(name, tool_profiles.get(name)) for name in tool_names}


def available_roles(classified: dict[str, set[ToolRole]]) -> set[ToolRole]:
    roles: set[ToolRole] = set()
    for tool_roles in classified.values():
        roles |= tool_roles
    return roles
