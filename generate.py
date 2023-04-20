import os
import sys
from PIL import Image
import replicate

def remove_background(image_path, output_path):
    # use replicate api to remove background
    # need to have REPLICATE_API_KEY environment variable set
    output = replicate.run(
    "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003",
    input={"image": open(image_path, "rb")}
    )
    # save output image
    output["image"].save(output_path)


def generate(
    # path to data directory
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'),
    # path to output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output'),
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
    generate()