import os

from PIL import Image, ImageDraw, ImageFont

from . import DATA_DIR, FONTS_DIR, OUTPUT_DIR


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