import os
import sys
from PIL import Image, ImageDraw, ImageFont
import replicate

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts')

def remove_background(image_path, output_path):
    # use replicate api to remove background
    # need to have REPLICATE_API_KEY environment variable set
    output = replicate.run(
        "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003",
        input={"image": open(image_path, "rb")}
    )
    # save output image
    output["image"].save(output_path)

def draw_text(
    image_path = os.path.join(DATA_DIR, 'test.png'), 
    output_path = os.path.join(OUTPUT_DIR, 'test.png'), 
    text = 'Hello World',
    text_color = (0, 0, 255), 
    font = 'hupo',
    font_size = 32,
    rectangle_color = (255, 255, 255),
    rectangle_padding = 5,
):
    # choose file based on font name from font dir
    font_path = os.path.join(FONTS_DIR, font + '.ttf')
    font = ImageFont.truetype(font_path, font_size)
    # open image
    image = Image.open(image_path)
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
    # save output image
    image.save(output_path)

def generate(
    # path to data directory
    data_dir = DATA_DIR,
    # path to output directory
    output_dir = OUTPUT_DIR,
    # name of directory with background images
    background_dir = 'background',
    # name of directory with foreground images
    foreground_dir = 'foreground',
    # output image size,
    size = (1280, 720),
):
    # create output directory if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # get list of background images
    background_images = os.listdir(os.path.join(data_dir, background_dir))
    # get list of foreground images
    foreground_images = os.listdir(os.path.join(data_dir, foreground_dir))
    # iterate over background images
    for background_image in background_images:
        # open background image
        background = Image.open(os.path.join(data_dir, background_dir, background_image))
        # resize background image
        background = background.resize(size)
        # iterate over foreground images
        for foreground_image in foreground_images:
            # open foreground image
            foreground = Image.open(os.path.join(data_dir, foreground_dir, foreground_image))
            # resize foreground image
            foreground = foreground.resize(size)
            # paste foreground image on background image
            background.paste(foreground, (0, 0), foreground)
            # save output image
            background.save(os.path.join(output_dir, background_image.replace('.jpg', '') + '_' + foreground_image))


if __name__ == '__main__':

    draw_text()
    # generate()