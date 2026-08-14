# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
import arxiv
import json
import os
import sys
from datetime import datetime, timedelta


class DailyArxivPipeline:
    def __init__(self):
        self.page_size = 100
        self.client = arxiv.Client(self.page_size)

    def process_item(self, item: dict, spider):
        item["pdf"] = f"https://arxiv.org/pdf/{item['id']}"
        item["abs"] = f"https://arxiv.org/abs/{item['id']}"
        search = arxiv.Search(
            id_list=[item["id"]],
        )
        paper = next(self.client.results(search))
        item["authors"] = [a.name for a in paper.authors]
        item["title"] = paper.title
        item["categories"] = paper.categories
        item["comment"] = paper.comment
        item["journal_ref"] = paper.journal_ref
        item["doi"] = paper.doi
        raw_authors = paper._raw.get("authors", [])
        item["author_affiliations"] = [
            {"author": author.name, "affiliation": raw_author.get("arxiv_affiliation")}
            for author, raw_author in zip(paper.authors, raw_authors)
            if raw_author.get("arxiv_affiliation")
        ]
        item["summary"] = paper.summary
        return item
