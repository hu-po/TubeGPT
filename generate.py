import os
import shutil
from io import BytesIO
from typing import Dict, List, Union

import numpy as np
import openai
import replicate
import requests
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
FONTS_DIR = os.path.join(ROOT_DIR, 'fonts')

# ChatGPT's favorite colors
COLORS = {
    "coral": (255, 127, 80),
    "turquoise": (64, 224, 208),
    "cornflower_blue": (100, 149, 237),
    "goldenrod": (218, 165, 32),
    "orchid": (218, 112, 214),
    "medium_purple": (147, 112, 219),
    "aquamarine": (127, 255, 212),
    "salmon": (250, 128, 114),
    "dark_olive_green": (85, 107, 47),
    "light_sea_green": (32, 178, 170),
}

# set replicate api token
with open(os.path.join(ROOT_DIR, 'replicate.txt'), 'r') as f:
    os.environ['REPLICATE_API_TOKEN'] = f.read()
# set openai api token
with open(os.path.join(ROOT_DIR, 'openai.txt'), 'r') as f:
    os.environ['OPENAI_API_KEY'] = f.read()

def gpt_text(
        prompt: Union[str, List[Dict[str, str]]],
        system: str = None,
        model: str = "gpt-3.5.turbo",
        temperature : float = 0.6,
        max_tokens: int = 32,
        top_p: float = 1,
        stop: List[str] = ["\n"],
):
    if isinstance(prompt, str):
        prompt = [{"role" : "user", "content" : prompt}]
    if system is not None:
        prompt.append({"role" : "system", "content" : system}) 
    response = openai.Completion.create(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=0,
        presence_penalty=0,
        stop=stop,
    )
    return response['choices'][0]['text']

def gpt_image(
    prompt: str = None,
    n: int = 1,
    image_path = os.path.join(DATA_DIR, 'test.png'),
    output_path = os.path.join(OUTPUT_DIR, 'test.png'),
    image_size: str = "1024x1024",
):
    if prompt is None:
        response = openai.Image.create_variation(
            image=open(image_path, "rb"),
            n=n,
            size=image_size,
        )
        img_url = response['data'][0]['url']
    else:
        response = openai.Image.create(
            prompt=prompt,
            n=n,
            size=image_size,
        )
        img_url = response['data'][0]['url']
    # save output image
    image = Image.open(BytesIO(requests.get(img_url).content))
    image.save(output_path)


def remove_bg(
    image_path = os.path.join(DATA_DIR, 'test.png'),
    output_path = os.path.join(OUTPUT_DIR, 'test_nobg.png'),
):
    # use replicate api to remove background
    # need to have REPLICATE_API_KEY environment variable set
    img_url = replicate.run(
        "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003",
        input={"image": open(image_path, "rb")}
    )
    # save output image
    image = Image.open(BytesIO(requests.get(img_url).content))
    image.save(output_path)

# TODO: Can you iterate on these variables directly from ChatGPT?
def draw_text(
    image : Image,
    text = 'Hello World',
    text_color = 'aquamarine', 
    font = 'hupo',
    font_size = 72,
    rectangle_color = 'dark_olive_green',
    rectangle_padding = 20,
):
    text_color = COLORS[text_color]
    rectangle_color = COLORS[rectangle_color]
    # choose file based on font name from font dir
    font_path = os.path.join(FONTS_DIR, font + '.ttf')
    font = ImageFont.truetype(font_path, font_size)
    # draw text on image
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
    draw.rectangle([rectangle_x1, rectangle_y1, rectangle_x2, rectangle_y2], fill=rectangle_color)
    
    # Render the text
    draw.text((x, y), text, fill=text_color, font=font)

    return image

def stack_fgbg(
    fg_image_path = os.path.join(DATA_DIR, 'bu.1.1.nobg', 'test_bu_nobg.png'),
    bg_image_path = os.path.join(DATA_DIR, 'bg.16.9', 'test_bg.png'),
    output_path = os.path.join(OUTPUT_DIR, 'test_nobg.png'),
    # path to output directory
    output_dir = OUTPUT_DIR,
    # output image size,
    bg_size = (1280, 720),
    fg_size = (420, 420),
    gaussian_mu_sig = ((0.05, 0.05), (0.5, 0.05)),
    **kwargs,
):
    # load images
    fg_image = Image.open(fg_image_path)
    bg_image = Image.open(bg_image_path)
    # resize images
    fg_image = fg_image.resize(fg_size)
    bg_image = bg_image.resize(bg_size)
    # Sample position for bu based on gaussian
    x = int(np.random.normal(size=1, loc=gaussian_mu_sig[0][0], scale=gaussian_mu_sig[0][1])[0] * bg_size[0])
    y = int(np.random.normal(size=1, loc=gaussian_mu_sig[1][0], scale=gaussian_mu_sig[1][1])[0] * bg_size[1])
    # paste images
    bg_image.paste(fg_image, (x, y), fg_image)
    # draw text on image
    bg_image = draw_text(bg_image, **kwargs)
    # save output image
    output_path = os.path.join(output_dir, fg_image_path)
    bg_image.save(output_path)

if __name__ == '__main__':

    # Empty output directory
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.mkdir(OUTPUT_DIR)

    # remove_background()
    # draw_text()
    stack_fgbg()