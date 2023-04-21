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
FONTS_DIR = os.path.join(ROOT_DIR, 'fonts')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')

# Empty output directory (opinionated code motherfucker)
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.mkdir(OUTPUT_DIR)

# set replicate api token
with open(os.path.join(ROOT_DIR, 'replicate.txt'), 'r') as f:
    os.environ['REPLICATE_API_TOKEN'] = f.read()
# set openai api token
with open(os.path.join(ROOT_DIR, 'openai.txt'), 'r') as f:
    _key = f.read()
    os.environ['OPENAI_API_KEY'] = _key
    openai.api_key = _key

def gpt_text(
        prompt: Union[str, List[Dict[str, str]]] = None,
        system: str = None,
        model: str = "gpt-3.5-turbo",
        temperature : float = 0.6,
        max_tokens: int = 32,
        stop: List[str] = ["\n"],
):
    if isinstance(prompt, str):
        prompt = [{"role" : "user", "content" : prompt}]
    elif prompt is None:
        prompt = []
    if system is not None:
        prompt.append({"role" : "system", "content" : system}) 
    response = openai.ChatCompletion.create(
        messages=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop,
    )
    return response['choices'][0]['message']['content']

def gpt_color():
    try:
        color_name = gpt_text(
            system=' '.join([
                "You generate unique and interesting colors for a crayon set.",
                "Crayon color names are only a few words.", 
                "Respond with the colors only: no extra text or explanations.",
            ]),
            temperature=0.99,
        )
        rgb = gpt_text(
            prompt=color_name,
            system=' '.join([
                "You generate RGB color tuples for digital art based on word descriptions.",
                "Respond with three integers in the range 0 to 255 representing R, G, and B.",
                "The three integers should be separated by commas, without spaces.",
                "Respond with the colors only: no extra text or explanations.",
            ]),
            temperature=0.1,
        )
        rgb = rgb.split(',')
        assert len(rgb) == 3
        rgb = tuple([int(x) for x in rgb])
        assert all([0 <= x <= 256 for x in rgb])
    except:
        color_name = 'black'
        rgb = (0, 0, 0)
    return rgb, color_name

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

def draw_text(
    image_path = os.path.join(DATA_DIR, 'test.png'),
    output_path = os.path.join(OUTPUT_DIR, 'test_text.png'),
    text = 'Hello World',
    text_color = (255, 255, 255),
    font = 'Exo2-Bold',
    font_size = 72,
    rectangle_color = (0, 0, 0),
    rectangle_padding = 20,
):
    # choose file based on font name from font dir
    font_path = os.path.join(FONTS_DIR, font + '.ttf')
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
    draw.rectangle([rectangle_x1, rectangle_y1, rectangle_x2, rectangle_y2], fill=rectangle_color)
    
    # Render the text
    draw.text((x, y), text, fill=text_color, font=font)
    image.save(output_path)

def resize_bg(
    image_path = os.path.join(DATA_DIR, 'example_graphs.png'),
    output_path = os.path.join(OUTPUT_DIR, 'example_graphs_resized.png'),
    canvas_size = (1280, 720),
):
    img = Image.open(image_path)
    # Keep aspect ratio, resize width to fit
    width, height = img.size
    new_width = canvas_size[0]
    new_height = int(height * new_width / width)
    resized_image = img.resize((new_width, new_height), Image.ANTIALIAS)

    # Create a new canvas with the desired size, transparent background
    canvas = Image.new('RGBA', canvas_size, (0, 0, 0, 0))

    # Center the resized image on the canvas
    paste_position = (
        int((canvas_size[0] - new_width) / 2),
        int((canvas_size[1] - new_height) / 2),
    )
    canvas.paste(resized_image, paste_position)

    # Save the result
    canvas.save(output_path)

def stack_fgbg(
    fg_image_path = os.path.join(DATA_DIR, 'bu.1.1.nobg', 'test_bu_nobg.png'),
    bg_image_path = os.path.join(DATA_DIR, 'bg.16.9', 'test_bg.png'),
    output_path = os.path.join(OUTPUT_DIR, 'test_nobg.png'),
    # output image size,
    bg_size = (1280, 720),
    fg_size = (420, 420),
    gaussian_mu_sig = ((0.05, 0.05), (0.5, 0.05)),
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
    # save output image
    bg_image.save(output_path)

if __name__ == '__main__':

    # remove_background()
    # draw_text()
    # stack_fgbg()
    # gpt_text()
    # for _ in range(5):
    #     print(gpt_color())
    resize_background()
