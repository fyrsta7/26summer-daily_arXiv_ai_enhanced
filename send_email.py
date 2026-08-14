#!/usr/bin/env python3
"""Send the complete daily arXiv digest through SMTP."""

import argparse
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--markdown")
    parser.add_argument("--raw-jsonl")
    parser.add_argument("--enhanced-jsonl")
    parser.add_argument("--filter-report")
    parser.add_argument("--no-new-content", action="store_true")
    return parser.parse_args()


def required_env(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return value


def main():
    args = parse_args()
    sender = required_env("EMAIL_FROM")
    recipient = required_env("EMAIL_TO")
    password = required_env("SMTP_PASSWORD")
    categories = os.environ.get("CATEGORIES", "cs.SE")

    if args.no_new_content:
        message = EmailMessage()
        message["From"] = sender
        message["To"] = recipient
        message["Subject"] = f"[Daily arXiv] {args.date} 今日无新论文"
        message.set_content(
            f"今日的 Daily arXiv 任务已正常运行。\n"
            f"在去重后没有发现新的 {categories} 论文，因此今天没有论文附件。\n\n"
            f"在线阅读：https://fyrsta7.github.io/26summer-daily_arXiv_ai_enhanced/\n"
        )

        with smtplib.SMTP_SSL("smtp.163.com", 465, timeout=60) as smtp:
            smtp.login(sender, password)
            refused = smtp.send_message(message)
            if refused:
                raise RuntimeError(f"SMTP refused recipients: {', '.join(refused)}")

        print(f"No-new-content email accepted by smtp.163.com for {recipient}")
        return

    attachment_args = [
        args.markdown,
        args.raw_jsonl,
        args.enhanced_jsonl,
        args.filter_report,
    ]
    if any(value is None for value in attachment_args):
        raise ValueError(
            "Digest emails require --markdown, --raw-jsonl, --enhanced-jsonl, "
            "and --filter-report"
        )

    paths = [Path(value) for value in attachment_args]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing email attachments: {', '.join(missing)}")

    markdown = paths[0].read_text(encoding="utf-8")
    paper_count = sum(1 for line in paths[2].read_text(encoding="utf-8").splitlines() if line.strip())

    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = f"[Daily arXiv] {args.date} {categories} 新论文（{paper_count}篇）"
    message.set_content(
        f"今日共收录 {paper_count} 篇 {categories} 论文。\n"
        f"在线阅读：https://fyrsta7.github.io/26summer-daily_arXiv_ai_enhanced/\n\n"
        f"{markdown}"
    )

    attachment_types = {
        ".md": ("text", "markdown"),
        ".jsonl": ("application", "json"),
        ".json": ("application", "json"),
    }
    for path in paths:
        maintype, subtype = attachment_types[path.suffix]
        message.add_attachment(
            path.read_bytes(), maintype=maintype, subtype=subtype, filename=path.name
        )

    with smtplib.SMTP_SSL("smtp.163.com", 465, timeout=60) as smtp:
        smtp.login(sender, password)
        refused = smtp.send_message(message)
        if refused:
            raise RuntimeError(f"SMTP refused recipients: {', '.join(refused)}")

    print(f"Email accepted by smtp.163.com for {recipient}; papers={paper_count}")


if __name__ == "__main__":
    main()
