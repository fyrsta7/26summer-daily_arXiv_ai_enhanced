import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sync_filtered_feishu_digests as sync  # noqa: E402


def test_markdown_paths_filters_to_date_named_daily_digests(tmp_path):
    for name in ("2026-08-01.md", "2026-08-02.md", "notes.md"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    args = type("Args", (), {"data_dir": str(tmp_path), "start_date": "2026-08-02", "end_date": None})()
    assert [path.name for path in sync.markdown_paths(args)] == ["2026-08-02.md"]
