import os
import shutil
import uuid
from typing import List

from tubegpt import DATA_DIR, OUTPUT_DIR
from tubegpt.google import (
    get_video_hashtags_from_description,
    get_video_info,
    get_video_sentence_from_description,
)
from tubegpt.pillow import draw_text, resize_bg, stack_fgbg
from tubegpt.openai import gpt_color, gpt_image, gpt_text
from tubegpt.replicate import remove_bg

desired_sentence = """In this stream we review the paper: "A Method for Animating Children's Drawings of the Human Figure" out of Meta AI Research.

https://paperswithcode.com/paper/a-method-for-automatically-animating-children
https://github.com/facebookresearch/AnimatedDrawings
"""
output_dir_name = "animated_drawings"
background_image = os.path.join(DATA_DIR, "2023-04-21_10-30.png")


def generate_thumbnails(
    input_image_path: str,
    output_tmp_dir: str,
    output_dir: str,
    title: str,
):
    fg_prompt = gpt_text(
        prompt="portrait of white bengal cat, blue eyes, cute, chubby",
        system=" ".join(
            [
                "You generate variations of string prompts for image generation.",
                "Respond with a single new variant of the user prompt. ",
                "Add several new interesting or related words.",
                "Respond with the prompt only: no extra text or explanations.",
            ]
        ),
    )

    # Foreground image remove background
    fg_img_name = str(uuid.uuid4())
    gpt_image(
        prompt=fg_prompt,
        output_path=os.path.join(output_tmp_dir, f"{fg_img_name}.png"),
        image_size="512x512",
    )
    remove_bg(
        image_path=os.path.join(output_tmp_dir, f"{fg_img_name}.png"),
        output_path=os.path.join(output_tmp_dir, f"{fg_img_name}_nobg.png"),
    )

    # Background image
    bg_img_name = str(uuid.uuid4())
    resize_bg(
        image_path=input_image_path,
        output_path=os.path.join(output_tmp_dir, f"{bg_img_name}.png"),
        canvas_size=(1280, 720),
    )

    text_rgb, text_color = gpt_color()
    rect_rgb, rect_color = gpt_color()

    # Write text on top of image
    draw_text(
        image_path=os.path.join(output_tmp_dir, f"{bg_img_name}.png"),
        output_path=os.path.join(output_tmp_dir, f"{bg_img_name}_text.png"),
        text=title,
        text_color=text_rgb,
        font_size=60,
        rectangle_color=rect_rgb,
        rectangle_padding=20,
    )

    # Stack foreground and background
    combo_image_name = str(uuid.uuid4())
    stack_fgbg(
        fg_image_path=os.path.join(output_tmp_dir, f"{fg_img_name}_nobg.png"),
        bg_image_path=os.path.join(output_tmp_dir, f"{bg_img_name}_text.png"),
        output_path=os.path.join(output_dir, f"{combo_image_name}.png"),
        bg_size=(1280, 720),
        fg_size=(420, 420),
    )


def generate_yttext(
    output_path: str,
    desired_sentence: str,
    example_video_ids: List[str],
):
    socials = """
Like 👍. Comment 💬. Subscribe 🟥.

⌨️ GitHub
https://github.com/hu-po

🗨️ Discord
https://discord.gg/XKgVSxB6dE

📸 Instagram
http://instagram.com/gnocchibengal
    """

    best_videos = []
    for video_id in example_video_ids:
        title, description = get_video_info(video_id)
        hashtags = get_video_hashtags_from_description(description)
        sentence = get_video_sentence_from_description(description)
        best_videos.append(
            {
                "title": title,
                "hashtags": hashtags,
                "sentence": sentence,
            }
        )

    in_context_titles = []
    for best_video in best_videos:
        in_context_titles += [{"role": "user", "content": best_video["sentence"]}]
        in_context_titles += [
            {
                "role": "assistant",
                "content": best_video["title"],
            }
        ]
    # Add the last part of the prompt
    in_context_titles += [
        {
            "role": "user",
            "content": desired_sentence,
        }
    ]
    title = gpt_text(
        prompt=in_context_titles,
        system=" ".join(
            [
                "You create titles for YouTube videos.",
                "Respond with the title that best fits the description provided by the user.",
                "Respond with the title only: no extra text or explanations.",
            ]
        ),
        temperature=0.6,
    )

    in_context_hashtags = []
    for best_video in best_videos:
        in_context_hashtags += [{"role": "user", "content": best_video["sentence"]}]
        in_context_hashtags += [
            {
                "role": "assistant",
                "content": best_video["hashtags"],
            }
        ]
    # Add the last part of the prompt
    in_context_hashtags += [
        {
            "role": "user",
            "content": desired_sentence,
        }
    ]
    hashtags = gpt_text(
        prompt=in_context_hashtags,
        system=" ".join(
            [
                "You create hashtags for YouTube videos.",
                "Respond with up to 4 hashtags that match the user prompt.",
                "Respond with the hashtags only: no extra text or explanations.",
            ]
        ),
        temperature=0.6,
    )

    # Combine all the parts together
    full_description = f"{title}\n{desired_sentence}\n{socials}\n{hashtags}"

    # Save the description
    description_path = os.path.join(output_path, f"{str(uuid.uuid4())}.txt")
    with open(description_path, "w") as f:
        f.write(full_description)

    return title


# Create output dir if it doesn't exist
output_dir = os.path.join(OUTPUT_DIR, output_dir_name)
output_tmp_dir = os.path.join(OUTPUT_DIR, output_dir_name, "tmp")
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
if os.path.exists(output_tmp_dir):
    shutil.rmtree(output_tmp_dir)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_tmp_dir, exist_ok=True)

title = generate_yttext(
    output_dir,
    desired_sentence,
    [
        "eMFfMz9uYlc",
        "vzGy6Att_Hk",
        "RLcr4bqGsEQ",
        "KSZiJ4k28b4",
        "8b6NhnNYtpg",
    ],
)
generate_thumbnails(
    background_image,
    output_tmp_dir,
    output_dir,
    title=title,
)
