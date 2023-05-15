import datetime
import logging
import os
import random
import re
import time
import uuid
from io import BytesIO
from typing import Dict, List, Union

import arxiv
import discord
import fitz  # PyMuPDF
import gradio as gr
import openai
import replicate
import requests
import torch
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from notion_client import Client
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tubegpt")
# Set formatting
formatter = logging.Formatter("📺|%(asctime)s|%(message)s")
# Set up console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
log.addHandler(ch)
# Set up file handler
fh = logging.FileHandler("tubegpt.log")
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
log.info(f"ROOT_DIR: {ROOT_DIR}")
KEYS_DIR = os.path.join(ROOT_DIR, ".keys")
log.info(f"KEYS_DIR: {KEYS_DIR}")
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
log.info(f"DATA_DIR: {DATA_DIR}")


def set_discord_key(key=None):
    if key is None:
        try:
            with open(os.path.join(KEYS_DIR, "discord.txt"), "r") as f:
                key = f.read()
        except FileNotFoundError:
            log.warning("Discord API key not found.")
            return
    os.environ["DISCORD_API_KEY"] = key
    log.info("Discord API key set.")


def set_replicate_key(key=None):
    if key is None:
        try:
            with open(os.path.join(KEYS_DIR, "replicate.txt"), "r") as f:
                key = f.read()
        except FileNotFoundError:
            log.warning("Replicate API key not found.")
            return
    os.environ["REPLICATE_API_TOKEN"] = key
    log.info("Replicate API key set.")


def set_openai_key(key=None):
    if key is None:
        try:
            with open(os.path.join(KEYS_DIR, "openai.txt"), "r") as f:
                key = f.read()
        except FileNotFoundError:
            log.warning("OpenAI API key not found.")
            return
    os.environ["OPENAI_API_KEY"] = key
    openai.api_key = key
    log.info("OpenAI API key set.")


def set_notion_key(key=None):
    if key is None:
        try:
            with open(os.path.join(KEYS_DIR, "notion.txt"), "r") as f:
                key = f.read()
        except FileNotFoundError:
            log.warning("Notion API key not found.")
            return
    os.environ["NOTION_API_KEY"] = key
    log.info("Notion API key set.")


def set_google_key(key=None):
    if key is None:
        try:
            with open(os.path.join(KEYS_DIR, "google.txt"), "r") as f:
                key = f.read()
        except FileNotFoundError:
            log.warning("Google API key not found.")
            return
    os.environ["GOOGLE_API_KEY"] = key
    log.info("Google API key set.")


def set_huggingface_key(key=None):
    if key is None:
        try:
            with open(os.path.join(KEYS_DIR, "huggingface.txt"), "r") as f:
                key = f.read()
        except FileNotFoundError:
            log.warning("HuggingFace API key not found.")
            return
    os.environ["HUGGINGFACE_API_KEY"] = key
    log.info("HuggingFace API key set.")


def segment_segformer(
    text,
    image,
    model_name="nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
):
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)

    image = Image.fromarray(image)
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

    # Predicted mask with class indices
    class_mask = outputs.logits.argmax(1)

    # Get all the unique classes, and their name
    class_ids = torch.unique(class_mask)
    class_names = [model.config.id2label[x.item()] for x in class_ids]

    if text in class_names:
        class_id = model.config.label2id[text]
        mask = class_mask == class_id
        mask = mask.squeeze().cpu().numpy()
        mask = mask.astype("uint8") * 255
        mask = Image.fromarray(mask)
        mask = mask.resize(image.size)

        # Apply mask to image
        image = Image.composite(image, mask, mask)
        return image, ",".join(class_names)
    return image, ",".join(class_names)


def find_paper(url: str) -> arxiv.Result:
    pattern = r"arxiv\.org\/(?:abs|pdf)\/(\d+\.\d+)"
    match = re.search(pattern, url)
    if match:
        arxiv_id = match.group(1)
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results())
        return paper
    else:
        return None


def find_repo(url: str) -> arxiv.Result:
    pattern = r"github.com/([^/]+)/([^/]+)"
    match = re.search(pattern, url)
    if match:
        usr = match.group(0)
        repo = match.group(1)
        return usr, repo
    else:
        return None


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


def create_notion_page(
    database_name: str,
    title: str,
    text: str,
    date: str,
):
    notion = Client(auth=os.environ["NOTION_API_KEY"])

    # Get the database
    query_filter = {"property": "object", "value": "database"}
    response = notion.search(query=database_name, filter=query_filter).get("results")
    database_id = response[0]["id"]

    # Create the pages
    _emoji = gpt_emoji(title)[0]
    notion.pages.create(
        parent={"database_id": database_id},
        icon={
            "type": "emoji",
            "emoji": _emoji,
        },
        properties={
            "Name": {
                "title": [
                    {
                        "text": {
                            "content": title,
                        }
                    }
                ]
            },
            "Date": {
                "date": {
                    "start": date,
                }
            },
            "Platform": {
                "multi_select": [
                    {"name": "YT Live"},
                ]
            },
            "Status": {"status": {"name": "Planning"}},
            "Type": {
                "multi_select": [
                    {"name": "Code"},
                    {"name": "Paper"},
                ]
            },
        },
    )

    # Sleep to let the page be created
    time.sleep(5)

    # Get the page
    query_filter = {"property": "object", "value": "page"}
    response = notion.search(query=title, filter=query_filter).get("results")
    page_id = response[0]["id"]

    # Add text to the page as a code block
    notion.blocks.children.append(
        block_id=page_id,
        children=[
            {
                "object": "block",
                "type": "code",
                "code": {
                    "language": "python",
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": text,
                            },
                        }
                    ],
                },
            }
        ],
    )


def gpt_text(
    prompt: Union[str, List[Dict[str, str]]] = None,
    system=None,
    model="gpt-3.5-turbo",
    temperature: float = 0.6,
    max_tokens: int = 32,
    stop: List[str] = ["\n"],
):
    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]
    elif prompt is None:
        prompt = []
    if system is not None:
        prompt = [{"role": "system", "content": system}] + prompt
    log.debug(f"Function call to GPT {model}: \n {prompt}")
    response = openai.ChatCompletion.create(
        messages=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop,
    )
    return response["choices"][0]["message"]["content"]


def gpt_emoji(prompt):
    try:
        emoji = gpt_text(
            prompt=prompt,
            system=" ".join(
                [
                    "Respond with a single emoji based on the user prompt.",
                    "Respond with only basic original emoji.",
                    "Respond with only one emoji.",
                    "Do not explain, respond with the single emoji only.",
                ]
            ),
            temperature=0.3,
        )
    except Exception:
        emoji = "👻"
    return emoji


def gpt_color():
    try:
        color_name = gpt_text(
            system=" ".join(
                [
                    "You generate unique and interesting colors for a crayon set.",
                    "Crayon color names are only a few words.",
                    "Respond with the colors only: no extra text or explanations.",
                ]
            ),
            temperature=0.99,
        )
        rgb = gpt_text(
            prompt=color_name,
            system=" ".join(
                [
                    "You generate RGB color tuples for digital art based on word descriptions.",
                    "Respond with three integers in the range 0 to 255 representing R, G, and B.",
                    "The three integers should be separated by commas, without spaces.",
                    "Respond with the colors only: no extra text or explanations.",
                ]
            ),
            temperature=0.1,
        )
        rgb = rgb.split(",")
        assert len(rgb) == 3
        rgb = tuple([int(x) for x in rgb])
        assert all([0 <= x <= 256 for x in rgb])
    except Exception:
        color_name = "black"
        rgb = (0, 0, 0)
    return rgb, color_name


def gpt_image(
    prompt: str,
    n: int = 1,
    image_size="512x512",
):
    log.debug(f"Image call to GPT with: \n {prompt}")
    response = openai.Image.create(
        prompt=prompt,
        n=n,
        size=image_size,
    )
    img_url = response["data"][0]["url"]
    image = Image.open(BytesIO(requests.get(img_url).content))
    # Output path for original image
    image_name = uuid.uuid4()
    image_path = os.path.join(DATA_DIR, f"{image_name}.png")
    image.save(image_path)
    return image_path


def draw_text(
    image_path=None,
    output_path=None,
    text="Hello World",
    text_color=(255, 255, 255),
    font_path=None,
    font_size=72,
    rectangle_color=(0, 0, 0),
    rectangle_padding=20,
    position_jitter=50,
):
    font = ImageFont.truetype(font_path, font_size)
    # draw text on image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    text_width, text_height = draw.textsize(text, font=font)
    # Calculate the position to center the text
    x = (image.size[0] - text_width) / 2
    y = (image.size[1] - text_height) / 2
    # Jitter position of Text
    x += random.randint(-position_jitter, position_jitter)
    y += random.randint(-position_jitter, position_jitter)
    # Draw a solid colored rectangle behind the text
    rectangle_x1 = x - rectangle_padding
    rectangle_y1 = y - rectangle_padding
    rectangle_x2 = x + text_width + 2 * rectangle_padding
    rectangle_y2 = y + text_height + 2 * rectangle_padding
    draw.rectangle(
        [rectangle_x1, rectangle_y1, rectangle_x2, rectangle_y2],
        fill=rectangle_color,
    )
    draw.text((x, y), text, fill=text_color, font=font)
    image.save(output_path)


def resize_bg(
    image=None,
    output_path=None,
    canvas_size=(1280, 720),
):
    image = Image.fromarray(image)
    # Keep aspect ratio, resize width to fit
    width, height = image.size
    new_width = canvas_size[0]
    new_height = int(height * new_width / width)
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)
    # Create a new canvas with the desired size, transparent background
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    # Center the resized image on the canvas
    paste_position = (
        int((canvas_size[0] - new_width) / 2),
        int((canvas_size[1] - new_height) / 2),
    )
    canvas.paste(resized_image, paste_position)
    canvas.save(output_path)


def stack_fgbg(
    fg_image=None,
    mask_image=None,
    bg_image=None,
    bg_image_path=None,
    output_path=None,
    bg_size=(1280, 720),
    fg_size=(420, 420),
    position_jitter=50,
):
    fg_image = Image.fromarray(fg_image)
    mask_image = Image.fromarray(mask_image)
    if bg_image_path is not None:
        bg_image = Image.open(bg_image_path)
    else:
        bg_image = Image.fromarray(bg_image)
    # resize images
    fg_image = fg_image.resize(fg_size)
    mask_image = mask_image.resize(fg_size)
    bg_image = bg_image.resize(bg_size)
    x, y = random.choice(
        [
            # Lower left corner
            (0, bg_size[1] - fg_size[1]),
            # Lower right corner
            (bg_size[0] - fg_size[0], bg_size[1] - fg_size[1]),
        ]
    )
    # Jitter x and y position
    x += random.randint(-position_jitter, position_jitter)
    y += random.randint(-position_jitter, position_jitter)
    # Final image
    image_full = Image.new("RGBA", bg_size)
    image_full.paste(fg_image, (x, y), mask_image)
    final = Image.alpha_composite(bg_image, image_full)
    final.save(output_path)


def remove_bg(
    image=None,
    image_path=None,
    output_path=None,
):
    # Temporary file location for image
    if image is not None:
        image_path = os.path.join(DATA_DIR, f"{uuid.uuid4()}.png")
        image = Image.fromarray(image)
        image.save(image_path)
    # use replicate api to remove background
    # need to have REPLICATE_API_KEY environment variable set
    img_url = replicate.run(
        "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003",
        input={"image": open(image_path, "rb")},
    )
    # save output image
    image = Image.open(BytesIO(requests.get(img_url).content))
    # Get the mask from image
    image_mask = image.split()[-1]
    if output_path is not None:
        image_mask.save(output_path)
    return image_mask


def get_video_sentence_from_description(description):
    # Split the text by the "Like" section
    parts = description.split("Like 👍.")

    # Get everything before the "Like" section
    text_before_like = parts[0].strip()
    return text_before_like


def repo_blurb(url: str) -> str:
    _ = find_repo(url)
    if _:
        usr, repo = _
        blurb = f"""
***** ⌨️ GitHub Repo *****
{url}
{usr} - {repo}
*************************
"""
        return blurb
    else:
        return None


def extract_images_from_pdf(
    pdf_url,
):
    # Download the pdf from the url
    pdf_path = os.path.join(DATA_DIR, f"{uuid.uuid4()}.pdf")
    with open(pdf_path, "wb") as f:
        f.write(requests.get(pdf_url).content)
    # List of images
    images = []
    # Open the PDF
    pdf = fitz.open(pdf_path)
    for page_num in range(pdf.page_count):
        page = pdf.load_page(page_num)
        image_list = page.get_images(full=True)
        for _, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf.extract_image(xref)
            image_bytes = base_image["image"]
            # Convert image bytes to PIL Image
            img = Image.open(BytesIO(image_bytes))
            # Convert PIL Image to numpy array
            images.append(img)
    # Close the PDF
    pdf.close()
    return images


def send_discord():
    intents = discord.Intents.default()
    intents.message_content = True
    client = MyClient(intents=intents)
    client.run(os.environ["DISCORD_API_KEY"])


class MyClient(discord.Client):
    async def on_ready(self):
        print("Logged on as", self.user)

    async def on_message(self, message):
        # don't respond to ourselves
        if message.author == self.user:
            return

        if message.content == "ping":
            await message.channel.send("pong")


def paper_blurb(paper: arxiv.Result) -> str:
    title = paper.title
    authors: List[str] = [author.name for author in paper.authors]
    published = paper.published.strftime("%m/%d/%Y")
    url = paper.pdf_url
    blurb = f"""
----- 📝 ArXiV -----
{url}
{title}
{published}
{", ".join(authors)}
--------------------
"""
    return blurb


def parse_textbox(text):
    references = ""
    hashtags = ""
    title = ""
    for url in gpt_text(
        prompt=" ".join(
            [
                "You find urls in text and return a comma separated list of clean urls.",
                "Do not explain, only return the list of urls. Here is the text: ",
                text,
            ],
        ),
    ).split(","):
        if find_paper(url):
            paper: arxiv.Result = find_paper(url)
            references += paper_blurb(paper)
            title += paper.title
            if "#arxiv" not in hashtags:
                hashtags += "#arxiv "
        if find_repo(url):
            references += repo_blurb(url)
            if "#github" not in hashtags:
                hashtags += "#github "
    return references, hashtags, title


def combine_texts(text, references, hashtags):
    return f"{text}{references}{hashtags}"


def generate_texts_title(title, max_tokens, temperature, model):
    return gpt_text(
        prompt=f"{title}",
        system=" ".join(
            [
                "Modify the given title for a YouTube video.",
                "Add or remove some words.",
                "Do not explain, answer with the title only.",
            ]
        ),
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
    )


def generate_texts_hashtags(title, hashtags, max_tokens, temperature, model):
    return gpt_text(
        prompt=hashtags,
        system=" ".join(
            [
                "Modify the given hashtags for a YouTube video.",
                "Add or remove some hashtags.",
                "There must be exactly 5 hastags.",
                "Do not explain, respond with the hashtags only.",
                f"The YouTube video is titled {title}",
            ]
        ),
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
    )


def generate_thumbnails(
    fg_image: str,
    fg_mask_image: str,
    bg_image: str,
    title: str,
    font_color: str,
    font_path: str,
    font_size: int,
    rect_color: str,
    rect_padding: int,
):
    image_name = str(uuid.uuid4())
    bg_image_path = os.path.join(DATA_DIR, f"{image_name}_bg.png")
    resize_bg(
        image=bg_image,
        output_path=bg_image_path,
    )
    # Convert hex color to rgb
    rect_color = tuple(int(rect_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    font_color = tuple(int(font_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    bg_image_text_path = os.path.join(DATA_DIR, f"{image_name}_text.png")
    draw_text(
        image_path=bg_image_path,
        output_path=bg_image_text_path,
        text=title,
        text_color=font_color,
        font_size=font_size,
        font_path=font_path.name,
        rectangle_color=rect_color,
        rectangle_padding=rect_padding,
    )
    image_path = os.path.join(DATA_DIR, f"{image_name}_final.png")
    stack_fgbg(
        fg_image=fg_image,
        mask_image=fg_mask_image,
        bg_image_path=bg_image_text_path,
        output_path=image_path,
    )
    return image_path


# Define the main GradIO UI
with gr.Blocks() as demo:
    gr.Markdown(
        """
# TubeGPT 📺

This AI tool helps you create YouTube videos. Start by finding a cool paper [Twitter List](https://twitter.com/i/lists/1653485531546767361), [PapersWithCode](https://paperswithcode.com/), [Reddit Feed](https://www.reddit.com/user/deephugs/m/ml/), [Arxiv Sanity](http://www.arxiv-sanity.com/)
"""
    )
    log.info("Starting GradIO Frontend ...")
    texts_text = gr.State(
        value="""Like 👍. Comment 💬. Subscribe 🟥.
🏘 Discord: https://discord.gg/XKgVSxB6dE
"""
    )
    texts_references = gr.State(value="")
    with gr.Tab("Texts"):
        # TODO: Scrape information from paper sources
        # TODO: List/recommend specific paper sources
        # TODO: Accept any text and then parse it.
        gr_input_textbox = gr.Textbox(
            placeholder="Paste text here (arxiv, github, ...)",
            show_label=False,
            lines=1,
        )
        with gr.Accordion(
            label="GPT Settings",
            open=False,
        ):
            gr_model = gr.Dropdown(
                choices=["gpt-3.5-turbo", "gpt-4"],
                label="GPT Model behind conversation",
                value="gpt-3.5-turbo",
            )
            gr_max_tokens = gr.Slider(
                minimum=1,
                maximum=300,
                value=50,
                label="max tokens",
                step=1,
            )
            gr_temperature = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.7,
                label="Temperature",
            )
        with gr.Row():
            gr_texts_title_button = gr.Button(value="Make Title")
            gr_texts_title_textbox = gr.Textbox(show_label=False)
            gr_texts_title_button.click(
                generate_texts_title,
                inputs=[
                    gr_texts_title_textbox,
                    gr_max_tokens,
                    gr_temperature,
                    gr_model,
                ],
                outputs=[gr_texts_title_textbox],
            )
        with gr.Row():
            gr_texts_hashtags_button = gr.Button(value="Make Hashtags")
            gr_texts_hashtags_textbox = gr.Textbox(show_label=False)
            gr_texts_hashtags_button.click(
                generate_texts_hashtags,
                inputs=[
                    gr_texts_title_textbox,
                    gr_texts_hashtags_textbox,
                    gr_max_tokens,
                    gr_temperature,
                    gr_model,
                ],
                outputs=[gr_texts_hashtags_textbox],
            )
        gr_input_textbox.change(
            parse_textbox,
            inputs=[gr_input_textbox],
            outputs=[
                texts_references,
                gr_texts_hashtags_textbox,
                gr_texts_title_textbox,
            ],
        )
        gr_generate_texts_button = gr.Button(value="Combine")
        gr_texts_textbox = gr.Textbox(label="Copy Paste into YouTube")
        gr_generate_texts_button.click(
            combine_texts,
            inputs=[
                texts_text,
                texts_references,
                gr_texts_hashtags_textbox,
            ],
            outputs=[gr_texts_textbox],
        )
    with gr.Tab("Thumbnail"):
        gr_bg_image = gr.Image(
            label="Background",
            image_mode="RGB",
        )
        with gr.Accordion(label="Extract Images from PDF", open=False):
            gr_extracted_images_gallery = gr.Gallery(
                label="Extracted Images",
                image_mode="RGB",
            )
            gr_extract_images_button = gr.Button(value="Extract Images")
            gr_extract_images_button.click(
                extract_images_from_pdf,
                inputs=[gr_input_textbox],
                outputs=[gr_extracted_images_gallery],
            )
        with gr.Accordion(label="Generate Foreground w/ OpenAI Image", open=False):
            with gr.Row():
                gr_fg_image = gr.Image(
                    label="Foreground",
                    image_mode="RGB",
                )
                with gr.Column():
                    gr_generate_fg_button = gr.Button(value="Generate Foreground")
                    gr_fg_prompt_textbox = gr.Textbox(
                        placeholder="Foreground Prompt",
                        show_label=False,
                        lines=1,
                        value="portrait of a blue eyed white bengal cat",
                    )
            gr_generate_fg_button.click(
                gpt_image,
                inputs=[gr_fg_prompt_textbox],
                outputs=[gr_fg_image],
            )
        with gr.Accordion(label="Remove Background with Replicate", open=False):
            with gr.Row():
                gr_mask_image = gr.Image(
                    label="Foreground Mask",
                    image_mode="L",
                )
                with gr.Column():
                    gr_make_mask_button = gr.Button(value="Make Mask")
            gr_make_mask_button.click(
                remove_bg,
                inputs=[gr_fg_image],
                outputs=[gr_mask_image],
            )
        with gr.Accordion(label="Segment with Segformer", open=False):
            with gr.Row():
                gr_segment_image = gr.Image(
                    label="Image to Segment",
                    image_mode="RGB",
                )
                with gr.Column():
                    gr_segment_button = gr.Button(value="Segment")
                    gr_segment_textbox = gr.Textbox(
                        placeholder="Class Name",
                        show_label=False,
                        lines=1,
                    )
                    gr_segment_out_textbox = gr.Textbox(
                        placeholder="Found Classes",
                        show_label=False,
                        lines=4,
                    )
            gr_segment_button.click(
                segment_segformer,
                inputs=[gr_segment_textbox, gr_segment_image],
                outputs=[gr_segment_image, gr_segment_out_textbox],
            )
        with gr.Row():
            gr_combine_button = gr.Button(value="Make Thumbnail")
            with gr.Accordion(
                label="Text Settings",
                open=False,
            ):
                with gr.Row():
                    gr_rect_color = gr.ColorPicker(
                        label="Rectangle Color",
                        value="#64dbf1",
                    )
                    gr_font_color = gr.ColorPicker(
                        label="Text Color",
                        value="#000000",
                    )
                gr_font_path = gr.File(
                    label="Font",
                    value=os.path.join(DATA_DIR, "RobotoMono-SemiBold.ttf"),
                )
                gr_font_size = gr.Slider(
                    minimum=50,
                    maximum=120,
                    value=92,
                    label="Font Size",
                    step=1,
                )
                gr_rect_padding = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=10,
                    label="Rectangle Padding",
                    step=1,
                )
        gr_combined_image = gr.Image(
            label="Combined",
            image_mode="RGB",
        )
        gr_combine_button.click(
            generate_thumbnails,
            inputs=[
                gr_fg_image,
                gr_mask_image,
                gr_bg_image,
                gr_texts_title_textbox,
                gr_font_color,
                gr_font_path,
                gr_font_size,
                gr_rect_color,
                gr_rect_padding,
            ],
            outputs=[gr_combined_image],
        )
    with gr.Tab("Notion"):
        notion_database_textbox = gr.Textbox(
            label="Notion Database Name",
            lines=1,
            value="Content Calendar",
        )
        gr_planned_date_textbox = gr.Textbox(
            label="Planned Date",
            lines=1,
            value=str(datetime.datetime.now().date()),
        )
        gr_export_notion_button = gr.Button(label="Export to Notion")
        gr_export_notion_button.click(
            create_notion_page,
            inputs=[
                notion_database_textbox,
                gr_texts_title_textbox,
                gr_texts_textbox,
                gr_planned_date_textbox,
            ],
        )

    with gr.Tab("Keys"):
        openai_api_key_textbox = gr.Textbox(
            placeholder="Paste your OpenAI API key here",
            show_label=False,
            lines=1,
            type="password",
        )
        openai_api_key_textbox.change(
            set_openai_key,
            inputs=[openai_api_key_textbox],
        )
        set_openai_key()
        replicate_api_key_textbox = gr.Textbox(
            placeholder="Paste your Replicate API key here",
            show_label=False,
            lines=1,
            type="password",
        )
        replicate_api_key_textbox.change(
            set_replicate_key,
            inputs=[replicate_api_key_textbox],
        )
        set_replicate_key()
        notion_api_key_textbox = gr.Textbox(
            placeholder="Paste your Notion API key here",
            show_label=False,
            lines=1,
            type="password",
        )
        notion_api_key_textbox.change(
            set_notion_key,
            inputs=[notion_api_key_textbox],
        )
        set_notion_key()
        huggingface_api_key_textbox = gr.Textbox(
            placeholder="Paste your HuggingFace API key here",
            show_label=False,
            lines=1,
            type="password",
        )
        huggingface_api_key_textbox.change(
            set_huggingface_key,
            inputs=[huggingface_api_key_textbox],
        )
        set_huggingface_key()

    gr.HTML(
        """
        <center>
        Author: <a href="https://youtube.com/@hu-po">Hu Po</a>
        GitHub: <a href="https://github.com/hu-po/TubeGPT">TubeGPT</a>
        <br>
        <a href="https://huggingface.co/spaces/hu-po/speech2speech?duplicate=true">
        <img src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
        </center>
        """
    )

if __name__ == "__main__":
    demo.launch()
