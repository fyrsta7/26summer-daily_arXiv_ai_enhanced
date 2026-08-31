import argparse
import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.publish_feishu_doc import (
    _parse_json_envelope,
    build_content,
    ensure_month_folder,
    normalize_digest_markdown,
    publish,
    run_lark,
)
from scripts.backfill_feishu_docs import select_dates


class PublishFeishuDocTests(unittest.TestCase):
    def test_parse_json_envelope_after_progress_text(self):
        value = _parse_json_envelope('Creating document...\n{"ok":true,"data":{"x":1}}\n')
        self.assertTrue(value["ok"])

    def test_normalize_removes_html_wrappers_but_keeps_paper_content(self):
        source = "<div id=toc></div>\n# Contents\n<details>\n<summary>Details</summary>\nMethod\x02\n</details>\n"
        normalized = normalize_digest_markdown(source)
        self.assertNotIn("<div", normalized)
        self.assertNotIn("<details>", normalized)
        self.assertIn("#### 详细信息", normalized)
        self.assertIn("Method", normalized)
        self.assertNotIn("\x02", normalized)

    def test_no_new_content_document(self):
        args = argparse.Namespace(
            date="2026-08-14",
            categories="cs.AI,cs.SE",
            no_new_content=True,
            markdown=None,
        )
        content = build_content(args)
        self.assertIn("2026-08-14", content)
        self.assertIn("没有发现新的 cs.AI,cs.SE 论文", content)

    @patch("scripts.publish_feishu_doc.run_lark")
    def test_existing_document_is_not_created_again(self, mocked_run):
        mocked_run.side_effect = [
            {
                "ok": True,
                "data": {
                    "files": [
                        {"type": "folder", "name": "2026-08", "token": "month-token"}
                    ],
                    "has_more": False,
                },
            },
            {
                "ok": True,
                "data": {
                    "files": [
                        {
                            "type": "docx",
                            "name": "Daily arXiv 2026-08-14",
                            "url": "https://example.feishu.cn/docx/existing",
                        }
                    ],
                    "has_more": False,
                },
            },
        ]
        args = argparse.Namespace(
            date="2026-08-14",
            folder_token="folder-token",
            categories="cs.AI,cs.SE",
            no_new_content=True,
            markdown=None,
            identity="user",
        )
        self.assertEqual(publish(args), "https://example.feishu.cn/docx/existing")
        self.assertEqual(mocked_run.call_count, 2)

    @patch("scripts.publish_feishu_doc.run_lark")
    def test_missing_month_folder_is_created(self, mocked_run):
        mocked_run.side_effect = [
            {"ok": True, "data": {"files": [], "has_more": False}},
            {
                "ok": True,
                "data": {
                    "folder_token": "new-month-token",
                    "url": "https://example.feishu.cn/drive/folder/new-month-token",
                },
            },
        ]
        folder = ensure_month_folder("root-token", "2026-09-01", "user")
        self.assertEqual(folder["token"], "new-month-token")
        self.assertEqual(folder["name"], "2026-09")

    @patch("scripts.publish_feishu_doc.subprocess.run")
    def test_run_lark_reports_structured_error(self, mocked_run):
        mocked_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr=json.dumps(
                {
                    "ok": False,
                    "error": {"message": "missing scope", "hint": "apply permission"},
                }
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "missing scope; apply permission"):
            run_lark(["docs", "+create"])

    def test_markdown_file_is_loaded(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "digest.md"
            path.write_text("# Digest\n", encoding="utf-8")
            args = argparse.Namespace(no_new_content=False, markdown=str(path))
            self.assertEqual(build_content(args), "# Digest\n")

    def test_backfill_date_range_is_inclusive(self):
        dates = ["2026-08-10", "2026-08-11", "2026-08-12"]
        self.assertEqual(
            select_dates(dates, "2026-08-11", "2026-08-12"),
            ["2026-08-11", "2026-08-12"],
        )


if __name__ == "__main__":
    unittest.main()
