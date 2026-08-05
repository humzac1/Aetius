from target_system.provenance import ArgumentProfile, ToolBehaviorProfile
from target_system.tool_roles import ToolRole, available_roles, classify_environment_tools, classify_tool_role


def _profile(tool_name, arg_names):
    return ToolBehaviorProfile(tool_name=tool_name, argument_profiles={name: ArgumentProfile() for name in arg_names})


# --- name-only classification (toy system's own tools, no profile) ---------


def test_send_email_classified_sensitive_action():
    assert classify_tool_role("send_email") == {ToolRole.SENSITIVE_ACTION}


def test_lookup_customer_classified_data_lookup():
    assert classify_tool_role("lookup_customer") == {ToolRole.DATA_LOOKUP}


def test_search_corpus_classified_entry_point():
    assert classify_tool_role("search_corpus") == {ToolRole.UNTRUSTED_CONTENT_ENTRY_POINT}


# --- real reconstructed-data regression cases -------------------------------
# these specific names caused false positives before tokenization (see
# target_system/tool_roles.py's module docstring) — locked in as regression tests


def test_create_invoice_gets_no_role():
    assert classify_tool_role("create_invoice") == set()


def test_calculate_total_gets_no_role():
    assert classify_tool_role("calculate_total") == set()


def test_send_invoice_gets_only_sensitive_action():
    assert classify_tool_role("send_invoice") == {ToolRole.SENSITIVE_ACTION}


def test_list_invoices_gets_only_data_lookup():
    assert classify_tool_role("list_invoices") == {ToolRole.DATA_LOOKUP}


def test_lookup_customer_with_invoice_id_arg_does_not_pick_up_sensitive_action():
    # "customer" contains "to" as a raw substring, "invoice_id" contains "id"
    # as a raw substring — both caused false positives before tokenization
    profile = _profile("lookup_customer", ["customer_id", "company_name"])
    assert classify_tool_role("lookup_customer", profile) == {ToolRole.DATA_LOOKUP}


def test_send_invoice_with_invoice_id_arg_does_not_pick_up_data_lookup():
    profile = _profile("send_invoice", ["invoice_id", "message"])
    assert classify_tool_role("send_invoice", profile) == {ToolRole.SENSITIVE_ACTION}


# --- argument-based signal (narrow, whole-token) -----------------------------


def test_recipient_argument_adds_sensitive_action():
    profile = _profile("dispatch", ["recipient", "body"])
    assert ToolRole.SENSITIVE_ACTION in classify_tool_role("dispatch", profile)


def test_identifier_argument_adds_data_lookup():
    profile = _profile("resolve", ["identifier"])
    assert ToolRole.DATA_LOOKUP in classify_tool_role("resolve", profile)


def test_generic_id_suffix_argument_does_not_add_data_lookup_alone():
    # "invoice_id" must not trigger data_lookup via the arg signal on its own
    profile = _profile("archive", ["invoice_id"])
    assert classify_tool_role("archive", profile) == set()


# --- environment-level aggregation ------------------------------------------


def test_classify_environment_tools_toy_system():
    classified = classify_environment_tools(["send_email", "lookup_customer", "search_corpus"])
    assert classified["send_email"] == {ToolRole.SENSITIVE_ACTION}
    assert classified["lookup_customer"] == {ToolRole.DATA_LOOKUP}
    assert classified["search_corpus"] == {ToolRole.UNTRUSTED_CONTENT_ENTRY_POINT}


def test_classify_environment_tools_empty_list():
    assert classify_environment_tools([]) == {}


def test_available_roles_unions_across_tools():
    classified = classify_environment_tools(["send_email", "lookup_customer", "calculate_total"])
    assert available_roles(classified) == {ToolRole.SENSITIVE_ACTION, ToolRole.DATA_LOOKUP}


def test_available_roles_empty_for_toolless_environment():
    assert available_roles({}) == set()
