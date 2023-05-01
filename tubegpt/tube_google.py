import os
import logging
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

log = logging.getLogger(__name__)


def set_google_key(key: str = None, keys_dir: str = None):
    if key is None:
        try:
            with open(os.path.join(keys_dir, "google.txt"), "r") as f:
                key = f.read()
        except FileNotFoundError:
            log.warning("Google API key not found. Some features may not work.")
    os.environ["GOOGLE_API_KEY"] = key
    log.info("Google API key set.")


def get_video_info(video_id):
    if not video_id:
        return None
    try:
        youtube = build("youtube", "v3", developerKey=os.environ["GOOGLE_API_KEY"])
        response = youtube.videos().list(part="snippet", id=video_id).execute()
        if "items" in response and len(response["items"]) > 0:
            description = response["items"][0]["snippet"]["description"]
            title = response["items"][0]["snippet"]["title"]
            return title, description
        else:
            return None
    except HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred: {e.content}")
        return None


def get_video_hashtags_from_description(description):
    # The last line of the description is the hashtags
    hashtags = description.splitlines()[-1]
    return hashtags


def get_video_sentence_from_description(description):
    # Split the text by the "Like" section
    parts = description.split("Like 👍.")

    # Get everything before the "Like" section
    text_before_like = parts[0].strip()
    return text_before_like
