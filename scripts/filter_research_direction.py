#!/usr/bin/env python3
"""Build an auditable collection of papers strongly related to code optimization."""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests


SYSTEM_PROMPT = """你是自动代码优化研究方向的资深论文审稿人。用户的代表工作 SemOpt 使用静态程序分析识别适用位置、从历史优化中提取和检索优化策略，并由 LLM 生成保持语义且能获得真实性能提升的代码。现在需要从已有论文中筛选与这一研究方向强相关的工作，而不是宽泛的 coding agent 推荐。

强相关包括：
1. LLM/agent 自动修改源代码、IR、汇编、kernel 或编译配置，以提升运行时间、吞吐、延迟、内存、能耗或代码尺寸，并验证正确性与真实性能；
2. 搜索、进化、强化学习、反馈驱动或多轮 agent 用于程序性能优化；
3. 编译器优化、superoptimization、autotuning、loop/vectorization、tensor/kernel 自动生成与优化，方法或反馈机制可直接迁移到 LLM 自动代码优化；
4. 专门评测自动代码性能优化的 benchmark、数据集、验证方法，或直接服务于自动优化的 profiling、热点定位、静态/动态分析、优化策略挖掘与检索。

必须排除：普通代码生成、修复、测试、审查或软件工程 agent（没有性能优化目标）；仅优化 LLM 推理系统、训练、量化、提示词或上下文；数学/组合优化求解；数据库查询/网络/调度/硬件设计优化；只讨论代码质量、安全性或正确性而没有性能优化；摘要里偶然出现 performance/optimization/code 但没有自动优化程序。

评分标准：
- 0.95-1.00：与 SemOpt 几乎同题，LLM/agent 自动进行程序性能优化并有执行反馈或 speedup 验证；
- 0.88-0.94：直接的自动程序/编译器/kernel 优化，或高度相关 benchmark；
- 0.82-0.87：可直接迁移的关键子问题，如面向自动优化的热点定位、策略检索或正确性/性能联合验证；
- 低于 0.82：不属于强相关。

只输出有效 JSON，不要 Markdown。必须对每个输入 id 恰好输出一次：
{"decisions":[{"id":"arxiv id","strongly_relevant":true,"score":0.0,"reason_zh":"一句具体理由","tags":["llm_code_optimization"]}]}
tags 只能从 llm_code_optimization、search_evolution、compiler_optimization、kernel_autotuning、benchmark_evaluation、profiling_analysis、optimization_retrieval 中选择。"""

STRICT_VERIFICATION_PROMPT = """你是自动代码性能优化领域的严格复核审稿人。下面的论文已通过宽召回初筛，现在必须排除“看起来可迁移、但研究对象并不是程序性能优化”的假阳性。

只有满足以下至少一项才可 strongly_relevant=true：
A. 核心任务是自动修改源代码、IR、汇编、编译pass/选项、GPU kernel、HLS/RTL代码，以改善运行时间、吞吐、延迟、内存、能耗或代码尺寸，并验证功能正确性与性能；
B. 核心贡献是专门面向自动程序性能优化的 benchmark、数据集、性能补丁语料或可靠 speedup 测量方法；
C. 核心贡献是为自动程序优化直接提供热点定位、profiling、静态/动态分析或优化策略挖掘/检索，而不是笼统的软件分析工具。

必须拒绝：游戏策略、控制策略、数学算法发现、科学发现、系统参数配置、模型训练/推理服务或硬件架构优化，即使产物以代码表示；普通代码生成、调试、修复、翻译、现代化或代码简洁度任务，除非“性能优化”是论文的明确核心目标且有真实性能验证；仅在一个附带实验中包含CUDA/kernel的通用方法；只是未来可能作为优化基础设施的工具。

评分：0.95以上为LLM/agent直接自动优化代码；0.88-0.94为直接编译器/kernel自动优化或专用benchmark；0.82-0.87只允许满足C的直接关键子问题。若不满足A/B/C，必须 strongly_relevant=false 且 score<0.82。

只输出有效 JSON，不要 Markdown。必须对每个输入 id 恰好输出一次：
{"decisions":[{"id":"arxiv id","strongly_relevant":true,"score":0.0,"reason_zh":"说明为何满足A/B/C或为何排除","tags":["llm_code_optimization"]}]}
tags 只能从 llm_code_optimization、search_evolution、compiler_optimization、kernel_autotuning、benchmark_evaluation、profiling_analysis、optimization_retrieval 中选择。"""

CODE_TERMS = re.compile(
    r"\b(code|program(?:s|ming)?|source|compiler|compilation|llvm|gcc|binary|assembly|"
    r"machine code|cuda|triton|kernel|loop|function|software|tensor operator|operator)\b",
    re.IGNORECASE,
)
PERFORMANCE_TERMS = re.compile(
    r"\b(optimi[sz](?:e|ed|es|ing|ation)|performance|speedup|run\s*time|runtime|latency|"
    r"throughput|memory|energy|efficien(?:t|cy)|profil(?:e|er|ing)|hotspot|auto.?tun(?:e|ing)|"
    r"superoptimi[sz]ation|vectori[sz](?:e|ation)|paralleli[sz](?:e|ation)|code size)\b",
    re.IGNORECASE,
)
DIRECT_TERMS = re.compile(
    r"(code|program|compiler|kernel|loop|llvm|cuda|triton).{0,50}"
    r"(performance|optimi[sz]|speedup|runtime|latency|throughput|auto.?tun)|"
    r"(performance|optimi[sz]|speedup|runtime|auto.?tun).{0,50}"
    r"(code|program|compiler|kernel|loop|llvm|cuda|triton)",
    re.IGNORECASE | re.DOTALL,
)
ALLOWED_TAGS = {
    "llm_code_optimization",
    "search_evolution",
    "compiler_optimization",
    "kernel_autotuning",
    "benchmark_evaluation",
    "profiling_analysis",
    "optimization_retrieval",
}
TAG_LABELS = {
    "llm_code_optimization": "LLM/Agent 代码优化",
    "search_evolution": "搜索与进化优化",
    "compiler_optimization": "编译器优化",
    "kernel_autotuning": "Kernel/自动调优",
    "benchmark_evaluation": "Benchmark/评测",
    "profiling_analysis": "Profiling/程序分析",
    "optimization_retrieval": "优化策略检索",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--threshold", type=float, default=0.82)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--supplemental-jsonl", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def inclusive_dates(start, end):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def is_candidate(item):
    text = f"{item.get('title', '')}\n{item.get('summary', '')}"
    return bool(DIRECT_TERMS.search(text) or (CODE_TERMS.search(text) and PERFORMANCE_TERMS.search(text)))


def load_unique_papers(data_dir, start, end, supplemental_paths=()):
    by_id = {}
    input_records = 0
    date_files = 0
    missing_dates = []
    for current in inclusive_dates(start, end):
        date_text = current.isoformat()
        path = data_dir / f"{date_text}_AI_enhanced_Chinese.jsonl"
        if not path.is_file():
            missing_dates.append(date_text)
            continue
        date_files += 1
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            input_records += 1
            item = json.loads(line)
            paper_id = str(item["id"])
            if paper_id not in by_id:
                item["source_dates"] = [date_text]
                by_id[paper_id] = item
            else:
                dates = by_id[paper_id]["source_dates"]
                if date_text not in dates:
                    dates.append(date_text)
                # Prefer the latest copy because daily enrichment may improve over time.
                latest = item
                latest["source_dates"] = dates
                by_id[paper_id] = latest
    supplemental_records = 0
    for supplemental_path in supplemental_paths:
        path = Path(supplemental_path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            supplemental_records += 1
            item = json.loads(line)
            paper_id = str(item["id"])
            if paper_id not in by_id:
                by_id[paper_id] = item
                continue
            existing = by_id[paper_id]
            existing["source_dates"] = list(
                dict.fromkeys(existing.get("source_dates", []) + item.get("source_dates", []))
            )
            if item.get("candidate_queries"):
                existing["candidate_queries"] = list(
                    dict.fromkeys(
                        existing.get("candidate_queries", []) + item["candidate_queries"]
                    )
                )
    return list(by_id.values()), input_records, date_files, missing_dates, supplemental_records


def request_decisions(batch, model, base_url, api_key, system_prompt=SYSTEM_PROMPT):
    papers = [
        {
            "id": item["id"],
            "title": item.get("title", ""),
            "abstract": item.get("summary", "")[:2200],
            "tldr_zh": item.get("AI", {}).get("tldr", "")[:500],
            "categories": item.get("categories", []),
        }
        for item in batch
    ]
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps({"papers": papers}, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    last_error = None
    for attempt in range(4):
        try:
            response = requests.post(
                base_url.rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=240,
            )
            response.raise_for_status()
            body = response.json()
            decisions = json.loads(body["choices"][0]["message"]["content"])["decisions"]
            expected = {paper["id"] for paper in papers}
            returned = [str(decision["id"]) for decision in decisions]
            if (set(returned) - expected) or len(returned) != len(set(returned)):
                raise ValueError(f"Unexpected or duplicate ids: expected={expected}, returned={returned}")
            usage = {
                key: int(body.get("usage", {}).get(key, 0) or 0)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            }
            missing = expected - set(returned)
            if missing:
                if len(batch) == 1:
                    raise ValueError(f"Model omitted singleton id: {next(iter(missing))}")
                for item in batch:
                    if item["id"] not in missing:
                        continue
                    recovered, recovered_usage = request_decisions(
                        [item], model, base_url, api_key, system_prompt
                    )
                    decisions.extend(recovered)
                    for key in usage:
                        usage[key] += recovered_usage[key]
            return decisions, usage
        except Exception as error:
            last_error = error
            if attempt < 3:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Research relevance request failed after 4 attempts: {last_error}")


def evaluate_candidates(
    candidates,
    model,
    base_url,
    api_key,
    batch_size,
    max_workers,
    system_prompt=SYSTEM_PROMPT,
    label="research filter",
):
    batches = [candidates[start:start + batch_size] for start in range(0, len(candidates), batch_size)]
    decisions = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                request_decisions, batch, model, base_url, api_key, system_prompt
            ): index
            for index, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            batch_decisions, batch_usage = future.result()
            decisions.extend(batch_decisions)
            for key in usage:
                usage[key] += batch_usage[key]
            print(
                f"Completed {label} batch {futures[future] + 1}/{len(batches)}",
                flush=True,
            )
    return decisions, usage


def normalize_decision(decision):
    score = max(0.0, min(1.0, float(decision.get("score", 0))))
    tags = [str(tag) for tag in decision.get("tags", []) if str(tag) in ALLOWED_TAGS]
    relevant_value = decision.get("strongly_relevant", False)
    strongly_relevant = relevant_value is True or str(relevant_value).lower() == "true"
    return {
        "id": str(decision["id"]),
        "strongly_relevant": strongly_relevant,
        "score": score,
        "reason_zh": str(decision.get("reason_zh", "")),
        "tags": list(dict.fromkeys(tags)),
    }


def markdown_for(selected, report):
    generated = report["generated_at"]
    lines = [
        "# 自动代码优化强相关论文精选",
        "",
        f"> 数据范围：{report['start_date']} 至 {report['end_date']}；生成时间：{generated}。",
        "> 以 SemOpt 的“程序分析/策略知识 + LLM 自动改写 + 正确性与真实性能验证”为研究锚点，只保留强相关论文。",
        "",
        f"共筛选 **{report['selected_total']}** 篇（去重候选论文 {report['unique_papers']} 篇，模型评估 {report['heuristic_candidates']} 篇）。",
        "",
        "## 快速索引",
        "",
    ]
    for index, item in enumerate(selected, 1):
        score = item["research_relevance"]["score"]
        lines.append(f"{index}. [{item.get('title', 'Untitled')}](#{index}) — {score:.2f}")
    lines.extend(["", "---", ""])
    for index, item in enumerate(selected, 1):
        relevance = item["research_relevance"]
        ai = item.get("AI", {})
        tag_text = "、".join(TAG_LABELS.get(tag, tag) for tag in relevance.get("tags", [])) or "自动代码优化"
        authors = ", ".join(item.get("authors", []))
        source_dates = ", ".join(item.get("source_dates", []))
        lines.extend(
            [
                f"<a id=\"{index}\"></a>",
                f"## {index}. [{item.get('title', 'Untitled')}]({item.get('abs', '')})",
                "",
                f"- **相关度**：{relevance['score']:.2f}",
                f"- **方向标签**：{tag_text}",
                f"- **收录日期**：{source_dates}",
                f"- **arXiv ID**：{item.get('id', '')}",
                f"- **作者**：{authors}",
                f"- **入选理由**：{relevance.get('reason_zh', '')}",
                "",
            ]
        )
        if ai.get("tldr"):
            lines.extend([f"**TL;DR**：{ai['tldr']}", ""])
        if ai.get("abstract_zh"):
            lines.extend([f"**中文摘要**：{ai['abstract_zh']}", ""])
        elif item.get("summary"):
            lines.extend([f"**原始摘要**：{item['summary']}", ""])
        if ai.get("method"):
            lines.extend([f"**方法**：{ai['method']}", ""])
        if ai.get("result"):
            lines.extend([f"**结果**：{ai['result']}", ""])
        lines.extend(["[返回索引](#快速索引)", "", "---", ""])
    return "\n".join(lines).rstrip() + "\n"


def main():
    args = parse_args()
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    if end < start:
        raise ValueError("end-date must not precede start-date")
    data_dir = Path(args.data_dir)
    papers, input_records, date_files, missing_dates, supplemental_records = load_unique_papers(
        data_dir, start, end, args.supplemental_jsonl
    )
    candidates = [item for item in papers if is_candidate(item)]
    print(
        json.dumps(
            {
                "date_files": date_files,
                "missing_dates": missing_dates,
                "input_records": input_records,
                "supplemental_records": supplemental_records,
                "unique_papers": len(papers),
                "heuristic_candidates": len(candidates),
            },
            ensure_ascii=False,
        )
    )
    if args.dry_run:
        return

    model = os.environ["MODEL_NAME"]
    decisions, first_pass_usage = evaluate_candidates(
        candidates,
        model,
        os.environ["OPENAI_BASE_URL"],
        os.environ["OPENAI_API_KEY"],
        args.batch_size,
        args.max_workers,
    )
    normalized = {decision["id"]: normalize_decision(decision) for decision in decisions}
    preliminary = []
    for item in candidates:
        decision = normalized[item["id"]]
        if decision["strongly_relevant"] and decision["score"] >= args.threshold:
            preliminary.append(item)

    verification_decisions, verification_usage = evaluate_candidates(
        preliminary,
        model,
        os.environ["OPENAI_BASE_URL"],
        os.environ["OPENAI_API_KEY"],
        min(args.batch_size, 12),
        args.max_workers,
        STRICT_VERIFICATION_PROMPT,
        "strict verification",
    )
    verified = {
        decision["id"]: normalize_decision(decision)
        for decision in verification_decisions
    }
    selected = []
    for item in preliminary:
        decision = verified[item["id"]]
        normalized[item["id"]]["verification"] = decision
        if decision["strongly_relevant"] and decision["score"] >= args.threshold:
            item["research_relevance"] = {
                "policy": "semopt_strong_relevance",
                "first_pass_score": normalized[item["id"]]["score"],
                **{key: decision[key] for key in ("score", "reason_zh", "tags")},
            }
            selected.append(item)
    selected.sort(
        key=lambda item: (
            item["research_relevance"]["score"],
            max(item.get("source_dates", [""])),
            item.get("id", ""),
        ),
        reverse=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    total_usage = {
        key: first_pass_usage[key] + verification_usage[key]
        for key in first_pass_usage
    }
    report = {
        "model": model,
        "policy": "semopt_strong_relevance",
        "threshold": args.threshold,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "generated_at": generated_at,
        "date_files": date_files,
        "missing_dates": missing_dates,
        "input_records": input_records,
        "supplemental_records": supplemental_records,
        "unique_papers": len(papers),
        "heuristic_candidates": len(candidates),
        "first_pass_selected": len(preliminary),
        "selected_total": len(selected),
        "first_pass_usage": first_pass_usage,
        "verification_usage": verification_usage,
        "filter_usage": total_usage,
    }
    (output_dir / "papers.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in selected),
        encoding="utf-8",
    )
    ordered_decisions = sorted(normalized.values(), key=lambda item: item["id"])
    (output_dir / "decisions.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in ordered_decisions),
        encoding="utf-8",
    )
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(markdown_for(selected, report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
