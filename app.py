import logging
import os
import re
import uuid
from io import BytesIO
from typing import Dict, List, Union

import numpy as np
import arxiv
import discord
import gradio as gr
import openai
import replicate
import requests
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


def set_discord_key(key: str = None, keys_dir: str = None):
    if key is None:
        try:
            with open(os.path.join(keys_dir, "discord.txt"), "r") as f:
                key = f.read()
        except FileNotFoundError:
            log.warning("Discord API key not found. Some features may not work.")
    os.environ["DISCORD_API_KEY"] = key
    log.info("Discord API key set.")


def set_replicate_key(key: str = None, keys_dir: str = None):
    if key is None:
        try:
            with open(os.path.join(keys_dir, "replicate.txt"), "r") as f:
                key = f.read()
        except FileNotFoundError:
            log.warning("Replicate API key not found. Some features may not work.")
    os.environ["REPLICATE_API_TOKEN"] = key
    log.info("Replicate API key set.")


def set_openai_key(key: str = None, keys_dir: str = None):
    if key is None:
        try:
            with open(os.path.join(keys_dir, "openai.txt"), "r") as f:
                key = f.read()
        except FileNotFoundError:
            log.warning("OpenAI API key not found. Some features may not work.")
    os.environ["OPENAI_API_KEY"] = key
    openai.api_key = key
    log.info("OpenAI API key set.")


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
    os.environ["NOTION_DATABASE_ID"] = database_id
    log.info("Notion database ID set.")


def set_google_key(key: str = None, keys_dir: str = None):
    if key is None:
        try:
            with open(os.path.join(keys_dir, "google.txt"), "r") as f:
                key = f.read()
        except FileNotFoundError:
            log.warning("Google API key not found. Some features may not work.")
    os.environ["GOOGLE_API_KEY"] = key
    log.info("Google API key set.")


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


def create_notion_page(paper):
    # Replace with the correct database ID
    notion = Client(auth=os.environ["NOTION_API_KEY"])
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

    notion.pages.create(
        parent={"database_id": os.environ["NOTION_DATABASE_ID"]}, properties=new_page
    )


def gpt_text(
    prompt: Union[str, List[Dict[str, str]]] = None,
    system: str = None,
    model: str = "gpt-3.5-turbo",
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
    data_dir: str,
    n: int = 1,
    image_size: str = "512x512",
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
    image_path = os.path.join(data_dir, f"{image_name}.png")
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
):
    font = ImageFont.truetype(font_path, font_size)
    # draw text on image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    text_width, text_height = draw.textsize(text, font=font)
    # Calculate the position to center the text
    x = (image.size[0] - text_width) / 2
    y = (image.size[1] - text_height) / 2

    # Draw a solid colored rectangle behind the text
    rectangle_x1 = x - rectangle_padding
    rectangle_y1 = y - rectangle_padding
    rectangle_x2 = x + text_width + rectangle_padding
    rectangle_y2 = y + text_height + rectangle_padding
    draw.rectangle(
        [rectangle_x1, rectangle_y1, rectangle_x2, rectangle_y2], fill=rectangle_color
    )

    # Render the text
    draw.text((x, y), text, fill=text_color, font=font)
    image.save(output_path)


def resize_bg(
    img=None,
    image_path=None,
    output_path=None,
    canvas_size=(1280, 720),
):
    if img is None:
        img = Image.open(image_path)
    else:
        img = Image.fromarray(img)
    # Keep aspect ratio, resize width to fit
    width, height = img.size
    new_width = canvas_size[0]
    new_height = int(height * new_width / width)
    resized_image = img.resize((new_width, new_height), Image.ANTIALIAS)

    # Create a new canvas with the desired size, transparent background
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))

    # Center the resized image on the canvas
    paste_position = (
        int((canvas_size[0] - new_width) / 2),
        int((canvas_size[1] - new_height) / 2),
    )
    canvas.paste(resized_image, paste_position)

    # Save the result
    canvas.save(output_path)


def stack_fgbg(
    fg_image: np.ndarray = None,
    bg_image: np.ndarray = None,
    fg_image_path: str=None,
    bg_image_path: str=None,
    output_path=None,
    bg_size=(1280, 720),
    fg_size=(420, 420),
):
    # load images
    if fg_image is None:
        fg_image = Image.open(fg_image_path)
    else:
        fg_image = Image.fromarray(fg_image)
    if bg_image is None:
        bg_image = Image.open(bg_image_path)
    else:
        bg_image = Image.fromarray(bg_image)
    # resize images
    fg_image = fg_image.resize(fg_size)
    bg_image = bg_image.resize(bg_size)
    # Add alpha channel to foreground
    fg_image = fg_image.convert("RGBA")
    # Upper left corner of the foreground such that it sits in the lower left corner of background
    x = 0
    y = bg_size[1] - fg_size[1]
    # Final image
    image_full = Image.new("RGBA", bg_size)
    image_full.paste(fg_image, (x, y), fg_image)
    final = Image.alpha_composite(bg_image, image_full)
    # paste images, account for alpha channel
    # save output image
    final.save(output_path)


def remove_bg(
    image: np.ndarray = None,
    data_dir: str = None,
    image_path: str=None,
    output_path: str=None,
):
    # Temporary file location for image
    if image is not None:
        image_path = os.path.join(data_dir, f"{uuid.uuid4()}.png")
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
    image.save(output_path)


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
        blurb: str = f"""
***** ⌨️ GitHub Repo *****
{url}
{usr} - {repo}
*************************
"""
        return blurb
    else:
        return None


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
    title: str = paper.title
    authors: List[str] = [author.name for author in paper.authors]
    published: str = paper.published.strftime("%m/%d/%Y")
    url: str = paper.pdf_url
    blurb: str = f"""
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


def generate_texts_title(new_title, cur_title, max_tokens, temperature, model):
    title = gpt_text(
        prompt=f"{new_title} {cur_title}",
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
    return title, title


def generate_texts_hashtags(
    cur_title, new_hashtags, cur_hashtags, max_tokens, temperature, model
):
    hashtags = gpt_text(
        prompt=f"{new_hashtags} {cur_hashtags}",
        system=" ".join(
            [
                "Modify the given hashtags for a YouTube video.",
                "Add or remove some hashtags.",
                "There must be exactly 5 hastags.",
                "Do not explain, respond with the hashtags only.",
                f"The YouTube video is titled {cur_title}",
            ]
        ),
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
    )
    return hashtags, hashtags


def generate_thumbnails(
    data_dir: str,
    gr_fg_image: str,
    gr_bg_image: str,
    title: str,
    font_path: str,
    font_size: int,
    rect_color: str,
    rect_padding: int,
):
    image_name = str(uuid.uuid4())
    # Remove foreground background with replicate
    image_nobg_path = os.path.join(data_dir, f"{image_name}_nobg.png")
    remove_bg(
        image=gr_fg_image,
        data_dir=data_dir,
        output_path=image_nobg_path,
    )
    # resize background
    resize_bg(
        img=gr_bg_image,
        output_path=os.path.join(data_dir, f"{image_name}.png"),
        canvas_size=(1280, 720),
    )
    # Convert hex color to rgb
    rect_color = tuple(int(rect_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    draw_text(
        image_path=os.path.join(data_dir, f"{image_name}.png"),
        output_path=os.path.join(data_dir, f"{image_name}_text.png"),
        text=title,
        text_color=(0, 0, 0),
        font_size=font_size,
        font_path=font_path,
        rectangle_color=rect_color,
        rectangle_padding=rect_padding,
    )
    image_path = os.path.join(data_dir, f"{image_name}_final.png")
    stack_fgbg(
        fg_image_path=image_nobg_path,
        bg_image_path=os.path.join(data_dir, f"{image_name}_text.png"),
        output_path=image_path,
        bg_size=(1280, 720),
        fg_size=(512, 512),
    )
    return image_path


# Define the main GradIO UI
with gr.Blocks() as demo:
    log.info("Starting GradIO Frontend ...")
    gr.HTML(
        """
        <center>
        <h1>TubeGPT📺</h1>
        </center>
    """
    )
    root_dir = gr.State(value=ROOT_DIR)
    keys_dir = gr.State(value=KEYS_DIR)
    data_dir = gr.State(value=DATA_DIR)
    texts_input = gr.State(value="")
    texts_title = gr.State(value="")
    texts_text = gr.State(
        value="""
Like 👍. Comment 💬. Subscribe 🟥.
🏘 Discord: https://discord.gg/XKgVSxB6dE
"""
    )
    texts_references = gr.State(value="")
    texts_hashtags = gr.State(value="")
    with gr.Tab("Texts"):
        # TODO: Scrape information from paper sources
        # TODO: List/recommend specific paper sources
        # TODO: Accept any text and then parse it.
        gr_input_textbox = gr.Textbox(
            placeholder="Paste text here (arxiv, github, ...)",
            show_label=False,
            lines=1,
        )
        gr_input_textbox.change(
            parse_textbox,
            inputs=[gr_input_textbox],
            outputs=[texts_references, texts_hashtags, texts_title],
        )
        with gr.Accordion(
            label="GPT Settings",
            open=False,
        ):
            gpt_model = gr.State("gpt-3.5-turbo")
            gpt_max_tokens = gr.State(value=50)
            gpt_temperature = gr.State(value=0.7)
            gr_model = gr.Dropdown(
                choices=["gpt-3.5-turbo", "gpt-4"],
                label="GPT Model behind conversation",
                value=gpt_model.value,
            )
            gr_model.update(gpt_model)
            gr_max_tokens = gr.Slider(
                minimum=1,
                maximum=300,
                value=gpt_max_tokens.value,
                label="max tokens",
                step=1,
            )
            gr_max_tokens.update(gpt_max_tokens)
            gr_temperature = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=gpt_temperature.value,
                label="Temperature",
            )
            gr_temperature.update(gpt_temperature)
        with gr.Row():
            gr_texts_title_button = gr.Button(value="Make Title")
            gr_texts_title_textbox = gr.Textbox(show_label=False)
            gr_texts_title_button.click(
                generate_texts_title,
                inputs=[
                    gr_texts_title_textbox,
                    texts_title,
                    gr_max_tokens,
                    gr_temperature,
                    gr_model,
                ],
                outputs=[gr_texts_title_textbox, texts_title],
            )
        with gr.Row():
            gr_texts_hashtags_button = gr.Button(value="Make Hashtags")
            gr_texts_hashtags_textbox = gr.Textbox(show_label=False)
            gr_texts_hashtags_button.click(
                generate_texts_hashtags,
                inputs=[
                    texts_title,
                    gr_texts_hashtags_textbox,
                    texts_hashtags,
                    gr_max_tokens,
                    gr_temperature,
                    gr_model,
                ],
                outputs=[gr_texts_hashtags_textbox, texts_hashtags],
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
        gr_data_dir_textbox = gr.Textbox(
            label="Local Data Directory",
            show_label=True,
            lines=1,
            value=data_dir.value,
        )
        gr_bg_image = gr.Image(
            label="Background",
            image_mode="RGB",
        )
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
                    value = "portrait of a blue eyed white bengal cat"
                )
        gr_generate_fg_button.click(
            gpt_image,
            inputs=[gr_fg_prompt_textbox, data_dir],
            outputs=[gr_fg_image],
        )
        with gr.Row():
            gr_combine_button = gr.Button(value="Combine")
            with gr.Accordion(
                label="Text Settings",
                open=False,
            ):
                rect_color = gr.State(value="#64dbf1")
                rect_padding = gr.State(value=20)
                font_path = gr.State(
                    value=os.path.join(data_dir.value, "RobotoMono-VariableFont_wght.ttf"))
                font_size = gr.State(value=72)

                gr_color_picker = gr.ColorPicker(
                    label="Rectangle Color",
                    value="#64dbf1",
                )
                gr_color_picker.update(rect_color)
                gr_font_path = gr.File(
                    label="Font",
                    accept=".ttf",
                    value=font_path.value,
                )
                gr_font_path.update(font_path)
                gr_font_size = gr.Slider(
                    minimum=50,
                    maximum=120,
                    value=font_size.value,
                    label="Font Size",
                    step=1,
                )
                gr_font_size.update(font_size)
                gr_rect_padding = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=rect_padding.value,
                    label="Rectangle Padding",
                    step=1,
                )
                gr_rect_padding.update(rect_padding)
        gr_combined_image = gr.Image(
            label="Combined",
            image_mode="RGB",
        )
        gr_combine_button.click(
            generate_thumbnails,
            inputs=[
                data_dir,
                gr_fg_image,
                gr_bg_image,
                texts_title,
                font_path,
                font_size,
                rect_color,
                rect_padding,
            ],
            outputs=[gr_combined_image],
        )

        # TODO: Jitter placement of forground/background/text

        # gr_desc_gallery = gr.Gallery(
        #     label="Generated images", show_label=False, elem_id="gallery"
        # ).style(columns=[2], rows=[2], object_fit="contain", height="auto")
        # gr_generate_button.click(generate_desc, None, gr_desc_gallery)

    with gr.Tab("Notion"):
        notion_database_id_textbox = gr.Textbox(
            placeholder="Paste your Notion database ID here",
            show_label=False,
            lines=1,
            type="password",
        )
        notion_database_id_textbox.change(
            set_database_id, notion_database_id_textbox, None
        )
        # TODO: Planned date with gpt fuzzy match
        gr_planned_date_textbox = gr.Textbox(
            placeholder="Paste the planned date here",
            show_label=False,
            lines=1,
        )
        gr_export_notion_button = gr.Button(label="Export to Notion")

    with gr.Tab("Keys"):
        openai_api_key_textbox = gr.Textbox(
            placeholder="Paste your OpenAI API key here",
            show_label=False,
            lines=1,
            type="password",
        )
        openai_api_key_textbox.change(
            set_openai_key,
            inputs=[openai_api_key_textbox, keys_dir],
        )
        set_openai_key(None, keys_dir.value)
        replicate_api_key_textbox = gr.Textbox(
            placeholder="Paste your Replicate API key here",
            show_label=False,
            lines=1,
            type="password",
        )
        replicate_api_key_textbox.change(
            set_replicate_key,
            inputs=[replicate_api_key_textbox, keys_dir],
        )
        set_replicate_key(None, keys_dir.value)

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
