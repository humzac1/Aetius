"""Where config/trace/run data lands, and the one-time migration that
brings pre-platformdirs data forward.

These matter because the previous locations were Path(__file__)-relative,
i.e. inside the installed package: a real pipx install wrote reconstructed
configs, 395MB of cached Braintrust traces and every run's JSONL into
.../pipx/venvs/caligula/lib/python3.14/site-packages/, invisible from any
source checkout and destroyed by the next reinstall.
"""

from pathlib import Path

from config import paths


def test_data_dirs_are_not_inside_the_installed_package():
    package_root = Path(paths.__file__).parent.parent
    for directory in (paths.CONFIGS_DIR, paths.RUNS_DIR, paths.TRACES_DIR, paths.TRACES_BRAINTRUST_DIR):
        assert package_root not in directory.parents, f"{directory} would be written into the package"


def test_data_dirs_are_absolute_so_they_do_not_follow_the_cwd():
    for directory in (paths.CONFIGS_DIR, paths.RUNS_DIR, paths.TRACES_DIR, paths.TRACES_BRAINTRUST_DIR):
        assert directory.is_absolute()


def test_data_dirs_all_live_under_one_data_dir():
    for directory in (paths.CONFIGS_DIR, paths.RUNS_DIR, paths.TRACES_DIR, paths.TRACES_BRAINTRUST_DIR):
        assert directory.parent == paths.DATA_DIR


def test_consumers_resolve_to_the_shared_locations():
    # The point of the move: every module that persists user data reads
    # the same constants, so nothing can quietly keep its own directory.
    from dashboard.data_access import RUNS_DIR as dashboard_runs
    from experiments.runner import DEFAULT_RUNS_DIR as experiments_runs
    from ingestion.braintrust_client import DEFAULT_TRACES_DIR as braintrust_traces
    from ingestion.langfuse_client import DEFAULT_TRACES_DIR as langfuse_traces
    from target_system.config import DEFAULT_CONFIGS_DIR
    from target_system.runner import DEFAULT_RUNS_DIR as target_runs

    assert DEFAULT_CONFIGS_DIR == paths.CONFIGS_DIR
    assert experiments_runs == target_runs == dashboard_runs == paths.RUNS_DIR
    assert langfuse_traces == paths.TRACES_DIR
    assert braintrust_traces == paths.TRACES_BRAINTRUST_DIR


def _migrate_between(monkeypatch, tmp_path: Path):
    legacy_configs, new_configs = tmp_path / "legacy" / "configs", tmp_path / "new" / "configs"
    monkeypatch.setattr(paths, "_MIGRATIONS", ((legacy_configs, new_configs),))
    return legacy_configs, new_configs


def test_migrate_copies_legacy_data_forward(monkeypatch, tmp_path: Path):
    legacy, new = _migrate_between(monkeypatch, tmp_path)
    legacy.mkdir(parents=True)
    (legacy / "cfg_abc.json").write_text('{"config_hash": "cfg_abc"}')
    (legacy / "nested").mkdir()
    (legacy / "nested" / "trace.json").write_text("[]")

    assert paths.migrate_legacy_data() == [(legacy, new)]
    assert (new / "cfg_abc.json").read_text() == '{"config_hash": "cfg_abc"}'
    assert (new / "nested" / "trace.json").read_text() == "[]"
    # Non-destructive: the legacy copy is left alone.
    assert (legacy / "cfg_abc.json").exists()


def test_migrate_never_overwrites_an_existing_file(monkeypatch, tmp_path: Path):
    legacy, new = _migrate_between(monkeypatch, tmp_path)
    legacy.mkdir(parents=True)
    new.mkdir(parents=True)
    (legacy / "cfg_abc.json").write_text("legacy")
    (new / "cfg_abc.json").write_text("current")
    (legacy / "cfg_def.json").write_text("only-in-legacy")

    paths.migrate_legacy_data()
    # Per-file, not per-directory: an already-populated destination still
    # picks up what it's missing, without clobbering what it has.
    assert (new / "cfg_abc.json").read_text() == "current"
    assert (new / "cfg_def.json").read_text() == "only-in-legacy"


def test_migrate_is_idempotent_and_reports_nothing_the_second_time(monkeypatch, tmp_path: Path):
    legacy, _ = _migrate_between(monkeypatch, tmp_path)
    legacy.mkdir(parents=True)
    (legacy / "cfg_abc.json").write_text("x")

    assert paths.migrate_legacy_data()
    assert paths.migrate_legacy_data() == []


def test_migrate_is_a_noop_when_there_is_nothing_legacy(monkeypatch, tmp_path: Path):
    _migrate_between(monkeypatch, tmp_path)  # neither directory exists
    assert paths.migrate_legacy_data() == []


# --- the caligula -> aetius rename migration ------------------------------------


def test_previous_appname_data_is_picked_up_on_fresh_launch(monkeypatch, tmp_path: Path):
    """Simulates a machine with a real pre-rename install: a populated
    caligula-named data dir (reconstructed configs, generated cases, trace
    caches, credentials .env) and no aetius-named dir yet. A fresh launch
    under the new name must copy everything forward — and never touch the
    source, never overwrite anything already at the destination."""
    old = tmp_path / "caligula"
    new = tmp_path / "aetius"
    (old / "configs").mkdir(parents=True)
    (old / "configs" / "cfg_real.json").write_text('{"config_hash": "cfg_real"}')
    (old / "generated_cases").mkdir()
    (old / "generated_cases" / "cfg_real.json").write_text('{"entries": []}')
    (old / "traces_braintrust").mkdir()
    (old / "traces_braintrust" / "trace1.json").write_text("[]")
    (old / ".env").write_text("ANTHROPIC_API_KEY=sk-real\n")

    monkeypatch.setattr(paths, "_MIGRATIONS", ((old, new),))
    migrated = paths.migrate_legacy_data()
    assert migrated == [(old, new)]

    # everything arrived
    assert (new / "configs" / "cfg_real.json").read_text() == '{"config_hash": "cfg_real"}'
    assert (new / "generated_cases" / "cfg_real.json").exists()
    assert (new / "traces_braintrust" / "trace1.json").exists()
    assert (new / ".env").read_text() == "ANTHROPIC_API_KEY=sk-real\n"
    # the old copy is left fully intact — copy-only, never delete
    assert (old / "configs" / "cfg_real.json").exists()
    assert (old / ".env").exists()

    # second launch: nothing new to do, and nothing gets overwritten
    (new / "configs" / "cfg_real.json").write_text('{"config_hash": "cfg_real", "edited": true}')
    assert paths.migrate_legacy_data() == []
    assert "edited" in (new / "configs" / "cfg_real.json").read_text()


def test_appname_migration_pairs_are_wired_into_the_real_migration_table():
    # The rename migration must run through the same startup call as the
    # pre-platformdirs one (tui.app.main -> migrate_legacy_data), so a
    # fresh aetius launch picks up caligula data without any manual step.
    sources = [src for src, _dest in paths._MIGRATIONS]
    assert paths.LEGACY_APPNAME_CONFIG_DIR in sources
    assert paths.APP_NAME == "aetius"
    assert paths.LEGACY_APPNAME_DATA_DIR.name == "caligula"


def test_legacy_data_dir_env_var_still_honored(monkeypatch):
    # Existing setups exporting CALIGULA_DATA_DIR must keep working; the
    # new AETIUS_DATA_DIR wins when both are set. Read-at-import means we
    # exercise the resolution logic directly rather than reloading.
    import os
    assert paths.DATA_DIR_ENV_VAR == "AETIUS_DATA_DIR"
    assert paths.LEGACY_DATA_DIR_ENV_VAR == "CALIGULA_DATA_DIR"
    resolve = lambda env: env.get("AETIUS_DATA_DIR") or env.get("CALIGULA_DATA_DIR")
    assert resolve({"CALIGULA_DATA_DIR": "/old"}) == "/old"
    assert resolve({"AETIUS_DATA_DIR": "/new", "CALIGULA_DATA_DIR": "/old"}) == "/new"
