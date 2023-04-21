import random
import os
import uuid
import pprint

from ops import (OUTPUT_DIR, DATA_DIR, gpt_text, get_video_hashtags_from_description, get_video_sentence_from_description, get_video_info)

desired_sentence = '''
In this stream we review the paper: "A Method for Animating Children's Drawings of the Human Figure" out of Meta AI Research.

https://paperswithcode.com/paper/a-method-for-automatically-animating-children
https://github.com/facebookresearch/AnimatedDrawings

'''

socials = '''
Like 👍. Comment 💬. Subscribe 🟥.

⌨️ GitHub
https://github.com/hu-po

🗨️ Discord
https://discord.gg/XKgVSxB6dE

📸 Instagram
http://instagram.com/gnocchibengal
'''

best_videos = []
for video_id in [
    "eMFfMz9uYlc",
    "vzGy6Att_Hk",
    "RLcr4bqGsEQ",
    "KSZiJ4k28b4",
    "8b6NhnNYtpg",
]:
    title, description = get_video_info(video_id)
    hashtags = get_video_hashtags_from_description(description)
    sentence = get_video_sentence_from_description(description)
    best_videos.append({
        "title": title,
        "hashtags": hashtags,
        "sentence": sentence,
    })

for _ in range(10):
    in_context_titles = []
    for best_video in best_videos:
        in_context_titles += [{
            "role": "user",
            "content": best_video["sentence"]
        }]
        in_context_titles += [{
            "role": "assistant",
            "content": best_video["title"],
        }]
    # Add the last part of the prompt
    in_context_titles += [{
        "role": "user",
        "content": desired_sentence,
    }]
    title = gpt_text(
        prompt=in_context_titles,
        system=' '.join([
            "You create titles for YouTube videos.",
            "Respond with the title that best fits the description provided by the user.",
            "Respond with the title only: no extra text or explanations.",
        ]),
        temperature=0.6,
    )
    

    in_context_hashtags = []
    for best_video in best_videos:
        in_context_hashtags += [{
            "role": "user",
            "content": best_video["sentence"]
        }]
        in_context_hashtags += [{
            "role": "assistant",
            "content": best_video["hashtags"],
        }]
    # Add the last part of the prompt
    in_context_hashtags += [{
        "role": "user",
        "content": desired_sentence,
    }]
    hashtags = gpt_text(
        prompt=in_context_hashtags,
        system=' '.join([
            "You create hashtags for YouTube videos.",
            "Respond with up to 4 hashtags that match the user prompt.",
            "Respond with the hashtags only: no extra text or explanations.",
        ]),
        temperature=0.6,
    )

    # Combine all the parts together
    full_description = f'{title}\n{desired_sentence}\n{socials}\n{hashtags}'
    print(full_description)