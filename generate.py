import os
import sys
from PIL import Image, ImageDraw, ImageFont
import replicate
import requests
from io import BytesIO
import numpy as np

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

def remove_background(
    image_path = os.path.join(DATA_DIR, 'test.png'),
    output_path = os.path.join(OUTPUT_DIR, 'test.png'),
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
    text_color = COLORS['aquamarine'], 
    font = 'hupo',
    font_size = 72,
    rectangle_color = COLORS['dark_olive_green'],
    rectangle_padding = 20,
):
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

def combine(
    foreground_image_name = 'test_bu_nobg.png',
    background_image_name = 'test_bg.png',
    foreground_dir = os.path.join(DATA_DIR, 'bu.1.1.nobg'),
    background_dir = os.path.join(DATA_DIR, 'bg.16.9'),
    # path to output directory
    output_dir = OUTPUT_DIR,
    # output image size,
    size = (1280, 720),
    # bu size
    bu_size = (420, 420),
    # distribution of bu in image
    bu_gaussian = ((0.05, 0.05), (0.5, 0.05)),
):
    # load images
    foreground_image = Image.open(os.path.join(foreground_dir, foreground_image_name))
    background_image = Image.open(os.path.join(background_dir, background_image_name))
    # resize images
    foreground_image = foreground_image.resize(bu_size)
    background_image = background_image.resize(size)
    # Sample position for bu based on gaussain
    x = int(np.random.normal(size=1, loc=bu_gaussian[0][0], scale=bu_gaussian[0][1])[0] * size[0])
    y = int(np.random.normal(size=1, loc=bu_gaussian[1][0], scale=bu_gaussian[1][1])[0] * size[1])
    # paste images
    background_image.paste(foreground_image, (x, y), foreground_image)
    # draw text on image
    background_image = draw_text(background_image)
    # save output image
    output_path = os.path.join(output_dir, foreground_image_name)
    background_image.save(output_path)

if __name__ == '__main__':

    # remove_background()
    # draw_text()
    combine()