import os
import logging
import notion

log = logging.getLogger(__name__)

NOTION_DATABASE_ID = "your_database_id"

def set_notion_key(key: str = None, keys_dir: str = None):
    if key is None:
        try:
            with open(os.path.join(keys_dir, "notion.txt"), "r") as f:
                key = f.read()
        except FileNotFoundError:
            log.warning("Notion API key not found. Some features may not work.")
    os.environ["NOTION_API_KEY"] = key
    log.info("Notion API key set.")

def set_database_id(database_id: str):
    global NOTION_DATABASE_ID
    NOTION_DATABASE_ID = database_id
    log.info("Notion database ID set.")


def create_notion_page(paper):
    # Replace with the correct database ID
    global NOTION_DATABASE_ID
    new_page = {
        "Name": {"title": [{"text": {"content": paper.title}}]},
        "Authors": {
            "rich_text": [
                {
                    "text": {
                        "content": ", ".join([author.name for author in paper.authors])
                    }
                }
            ]
        },
        "arXiv ID": {
            "rich_text": [{"text": {"content": paper.entry_id.split("/")[-1]}}]
        },
        "Abstract": {"rich_text": [{"text": {"content": paper.summary}}]},
    }

    notion.pages.create(parent={"database_id": NOTION_DATABASE_ID}, properties=new_page)
