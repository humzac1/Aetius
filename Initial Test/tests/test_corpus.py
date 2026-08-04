from pathlib import Path

from target_system.tools import CallLog, run_context, search_corpus

CORPUS_DIR = Path(__file__).parent.parent / "target_system" / "corpus"


def test_corpus_has_thirty_files():
    files = list(CORPUS_DIR.glob("*.md"))
    assert len(files) == 30


def test_corpus_has_all_three_document_types():
    files = [p.name for p in CORPUS_DIR.glob("*.md")]
    assert sum(f.startswith("wiki_") for f in files) == 10
    assert sum(f.startswith("meeting_notes_") for f in files) == 10
    assert sum(f.startswith("ticket_") for f in files) == 10


def test_search_corpus_finds_relevant_document():
    cl = CallLog()
    with run_context(cl, corpus_dir=CORPUS_DIR):
        result = search_corpus.entrypoint(query="travel per diem policy", max_results=3)
    assert "wiki_travel_policy.md" in result["files"]
    assert "$50/day" in result["excerpts"]["wiki_travel_policy.md"]
