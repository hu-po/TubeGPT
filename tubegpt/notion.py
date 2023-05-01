import os

from notion_client import Client
from . import log, KEYS_DIR

try:
    with open(os.path.join(KEYS_DIR, 'notion.txt'), 'r') as f:
        _key = f.read()
        os.environ['NOTION_API_KEY'] = _key
        NOTION_API_KEY = _key
        notion = Client(auth=_key)
except FileNotFoundError:
    log.warning('Notion API key not found. Some features may not work.')
    

def create_notion_page(paper):
    # Replace with the correct database ID
    database_id = "your_database_id"

    new_page = {
        "Name": {
            "title": [
                {
                    "text": {
                        "content": paper.title
                    }
                }
            ]
        },
        "Authors": {
            "rich_text": [
                {
                    "text": {
                        "content": ', '.join([author.name for author in paper.authors])
                    }
                }
            ]
        },
        "arXiv ID": {
            "rich_text": [
                {
                    "text": {
                        "content": paper.entry_id.split('/')[-1]
                    }
                }
            ]
        },
        "Abstract": {
            "rich_text": [
                {
                    "text": {
                        "content": paper.summary
                    }
                }
            ]
        }
    }

    notion.pages.create(parent={"database_id": database_id}, properties=new_page)
