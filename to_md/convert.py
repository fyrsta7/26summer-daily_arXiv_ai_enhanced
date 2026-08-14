import json
import argparse
import os
from itertools import count


def build_publication_metadata(item):
    lines = []
    comment = str(item.get("comment") or "").strip()
    if comment:
        lines.append(f"Comments: {comment}")

    journal_ref = str(item.get("journal_ref") or "").strip()
    if journal_ref:
        lines.append(f"Journal reference: {journal_ref}")

    doi = str(item.get("doi") or "").strip()
    if doi:
        doi_url = doi if doi.startswith(("http://", "https://")) else f"https://doi.org/{doi}"
        lines.append(f"DOI: [{doi}]({doi_url})")

    author_affiliations = item.get("author_affiliations") or []
    affiliation_text = []
    for entry in author_affiliations:
        if not isinstance(entry, dict):
            continue
        author = str(entry.get("author") or "").strip()
        affiliation = str(entry.get("affiliation") or "").strip()
        if affiliation:
            affiliation_text.append(f"{author}: {affiliation}" if author else affiliation)
    if affiliation_text:
        lines.append("Author affiliations: " + "; ".join(affiliation_text))

    return "\n\n".join(lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to the jsonline file")
    args = parser.parse_args()
    data = []
    preference = os.environ.get('CATEGORIES', 'cs.CV, cs.CL').split(',')
    preference = list(map(lambda x: x.strip(), preference))
    def rank(cate):
        if cate in preference:
            return preference.index(cate)
        else:
            return len(preference)

    with open(args.data, "r") as f:
        for line in f:
            data.append(json.loads(line))

    categories = set([item["categories"][0] for item in data])
    template = open("paper_template.md", "r").read()
    categories = sorted(categories, key=rank)
    cnt = {cate: 0 for cate in categories}
    for item in data:
        if item["categories"][0] not in cnt.keys():
            continue
        cnt[item["categories"][0]] += 1

    markdown = f"<div id=toc></div>\n\n# Table of Contents\n\n"
    for idx, cate in enumerate(categories):
        markdown += f"- [{cate}](#{cate}) [Total: {cnt[cate]}]\n"

    idx = count(1)
    for cate in categories:
        markdown += f"\n\n<div id='{cate}'></div>\n\n"
        markdown += f"# {cate} [[Back]](#toc)\n\n"
        papers = []
        for item in data:
            if item["categories"][0] == cate:
                # Safely access AI fields with default values
                ai_data = item.get('AI', {})
                if not ai_data or not isinstance(ai_data, dict):
                    print(f"Skipping item '{item.get('title', 'Unknown')}' due to missing or invalid AI data")
                    continue
                
                # Check if all required AI fields are present
                required_fields = ['abstract_zh', 'tldr', 'motivation', 'method', 'result', 'conclusion']
                if not all(field in ai_data for field in required_fields):
                    print(f"Skipping item '{item.get('title', 'Unknown')}' due to incomplete AI fields")
                    continue
                
                papers.append(
                    template.format(
                        title=item["title"],
                        authors=",".join(item["authors"]),
                        summary=item["summary"],
                        abstract_zh=ai_data.get('abstract_zh', ''),
                        selection_reason=item.get('selection', {}).get('reason_zh', ''),
                        url=item['abs'],
                        tldr=ai_data.get('tldr', ''),
                        motivation=ai_data.get('motivation', ''),
                        method=ai_data.get('method', ''),
                        result=ai_data.get('result', ''),
                        conclusion=ai_data.get('conclusion', ''),
                        publication_metadata=build_publication_metadata(item),
                        cate=item['categories'][0],
                        idx=next(idx)
                    )
                )
        markdown += "\n\n".join(papers)
    with open(args.data.split('_')[0] + '.md', "w") as f:
        f.write(markdown)
