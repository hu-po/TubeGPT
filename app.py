import logging
import os
import re
import uuid
from io import BytesIO
from typing import Dict, List, Union

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

def set_discord_key(key: str = None, keys_dir: str = None):
    if key is None:
        try:
            with open(os.path.join(keys_dir, "discord.txt"), "r") as f:
                key = f.read()
        except FileNotFoundError:
            log.warning("Discord API key not found. Some features may not work.")
    os.environ["DISCORD_API_KEY"] = key
    log.info("Discord API key set.")

def find_repo(url: str) -> arxiv.Result:
    pattern = r"github.com/([^/]+)/([^/]+)"
    match = re.search(pattern, url)
    if match:
        usr = match.group(0)
        repo = match.group(1)
        return usr, repo
    else:
        return None

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
    prompt: str = None,
    n: int = 1,
    image_path=None,
    output_path=None,
    image_size: str = "1024x1024",
):
    if prompt is None:
        log.debug(f"Image variation call to GPT: \n {image_path}")
        response = openai.Image.create_variation(
            image=open(image_path, "rb"),
            n=n,
            size=image_size,
        )
        img_url = response["data"][0]["url"]
    else:
        log.debug(f"Image call to GPT with: \n {prompt}")
        response = openai.Image.create(
            prompt=prompt,
            n=n,
            size=image_size,
        )
        img_url = response["data"][0]["url"]
    image = Image.open(BytesIO(requests.get(img_url).content))
    # save output image
    image.save(output_path)
    return img_url, output_path

def draw_text(
    image_path=None,
    output_path=None,
    text="Hello World",
    text_color=(255, 255, 255),
    font="Exo2-Bold",
    font_size=72,
    font_dir=None,
    rectangle_color=(0, 0, 0),
    rectangle_padding=20,
):
    # choose file based on font name from font dir
    font_path = os.path.join(font_dir, font + ".ttf")
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
    image_path=None,
    output_path=None,
    canvas_size=(1280, 720),
):
    img = Image.open(image_path)
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
    fg_image_path=None,
    bg_image_path=None,
    output_path=None,
    # output image size,
    bg_size=(1280, 720),
    fg_size=(420, 420),
):
    # load images
    fg_image = Image.open(fg_image_path)
    bg_image = Image.open(bg_image_path)
    # resize images
    fg_image = fg_image.resize(fg_size)
    bg_image = bg_image.resize(bg_size)
    # Upper left corner of the foreground such that it sits in the lower left corner of background
    x = 0
    y = bg_size[1] - fg_size[1]
    # Final image
    fg_image_full = Image.new("RGBA", bg_size)
    fg_image_full.paste(fg_image, (x, y), fg_image)
    final = Image.alpha_composite(bg_image, fg_image_full)
    # paste images, account for alpha channel
    # save output image
    final.save(output_path)



log = logging.getLogger(__name__)


def set_replicate_key(key: str = None, keys_dir: str = None):
    if key is None:
        try:
            with open(os.path.join(keys_dir, "replicate.txt"), "r") as f:
                key = f.read()
        except FileNotFoundError:
            log.warning("Replicate API key not found. Some features may not work.")
    os.environ["REPLICATE_API_TOKEN"] = key
    log.info("Replicate API key set.")


def remove_bg(
    image_path=None,
    output_path=None,
):
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
***** Code *****
{repo}
{url}
****************
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


def paper_blurb(url: str) -> str:
    paper: arxiv.Result = find_paper(url)
    if paper:
        title: str = paper.title
        authors: List[str] = [author.name for author in paper.authors]
        published: str = paper.published.strftime("%m/%d/%Y")
        url: str = paper.pdf_url
        blurb: str = f"""
----- Paper -----
{title}
{", ".join(authors)}
Released on {published}
ArXiV: {url}
-----------------
        """
        return blurb
    else:
        return None


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def parse_textbox(text):
    cleantext = text.copy()
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
            cleantext += paper_blurb(url)
        if find_repo(url):
            cleantext += repo_blurb(url)
    return cleantext


def combine_texts(info, socials, title, hashtags):
    return f"{title}{info}{socials}{hashtags}"


def generate_texts_title(prompt, max_tokens, temperature, model):
    return gpt_text(
        prompt=prompt,
        system=" ".join(
            [
                "You create titles for YouTube videos.",
                "Respond with a short title that best fits the description.",
                "Respond with the title only: no extra text or explanations.",
            ]
        ),
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
    )


def generate_texts_hashtags(prompt, max_tokens, temperature, model):
    return gpt_text(
        prompt=prompt,
        system=" ".join(
            [
                "You create hashtags for YouTube videos.",
                "Respond with up to 4 hashtags that match the user prompt.",
                "Respond with the hashtags only: no extra text or explanations.",
            ]
        ),
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
    )


def generate_thumbnails(
    prompt: str,
    input_image_path: str,
    output_tmp_dir: str,
    output_dir: str,
    title: str,
    n: int = 1,
):
    fg_prompt = gpt_text(
        prompt=prompt,
        system=" ".join(
            [
                "You generate variations of string prompts for image generation.",
                f"Respond with {n} new variants of the user prompt. ",
                "Do not explain, respond with the prompts only.",
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


# Define the main GradIO UI
with gr.Blocks() as demo:
    gr.HTML(
        """
        <center>
        <h1>TubeGPT</h1>
        </center>
    """
    )
    log.info("Initializing TubeGPT ...")

    # Directories for images, temporary files, API keys, etc
    root_dir = gr.State(value=os.path.dirname(os.path.abspath(__file__)))
    log.info(f"Root directory: {root_dir.value}")
    keys_dir = gr.State(value=os.path.join(root_dir.value, ".keys"))
    log.info(f"Keys directory: {keys_dir.value}")
    fonts_dir = gr.State(value=os.path.join(root_dir.value, "fonts"))
    log.info(f"Fonts directory: {fonts_dir.value}")
    output_dir = gr.State(value=os.path.join(root_dir.value, "data"))
    log.info(f"Output directory: {output_dir.value}")

    # Texts (Title, Descriptions, etc)
    texts_input = gr.State(value="")
    texts_urls = gr.State(value="")
    texts_socials = gr.State(
        value="""
Like 👍. Comment 💬. Subscribe 🟥.

⌨️ GitHub
https://github.com/hu-po

🗨️ Discord
https://discord.gg/XKgVSxB6dE

📸 Instagram
http://instagram.com/gnocchibengal

"""
    )
    with gr.Tab("Texts"):
        # TODO: Accept any text and then parse it.
        gr_input_textbox = gr.Textbox(
            placeholder="Paste the arXiv link here",
            show_label=False,
            lines=1,
            value=texts_input.value,
        )
        gr_input_textbox.update(texts_input)
        gr_parse_text_button = gr.Button(label="Parse Text")
        gr_parse_text_button.click(
            parse_textbox,
            inputs=[gr_input_textbox],
            outputs=[texts_urls],
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
                inputs=[texts_input, gr_max_tokens, gr_temperature, gr_model],
                outputs=[gr_texts_title_textbox],
            )
        with gr.Row():
            gr_texts_hashtags_button = gr.Button(value="Make Hashtags")
            gr_texts_hashtags_textbox = gr.Textbox(show_label=False)
            gr_texts_hashtags_button.click(
                generate_texts_hashtags,
                inputs=[texts_input, gr_max_tokens, gr_temperature, gr_model],
                outputs=[gr_texts_hashtags_textbox],
            )
        gr_generate_texts_button = gr.Button(value="Combine")
        gr_texts_textbox = gr.Textbox(label="Copy Paste into YouTube")
        gr_generate_texts_button.click(
            combine_texts,
            inputs=[
                texts_input,
                texts_socials,
                gr_texts_title_textbox,
                gr_texts_hashtags_textbox,
            ],
            outputs=[gr_texts_textbox],
        )
    with gr.Tab("Thumbnail"):
        with gr.Row():
            gr_generate_fg_button = gr.Button(label="Generate Foreground")
            gr_fg_prompt_textbox = gr.Textbox(
                placeholder="portrait of white bengal cat with blue eyes",
                show_label=False,
                lines=1,
            )
        gr_fg_image = gr.Image(
            label="Foreground",
            image_mode="RGB",
        )
        gr_generate_fg_button.click(
            gpt_image,
            inputs=[gr_fg_prompt_textbox, output_dir],
            outputs=[gr_fg_image],
        )
        # Add background image
        gr_bg_image = gr.Image(
            label="Background",
            image_mode="RGB",
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
