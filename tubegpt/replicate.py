import os
from io import BytesIO

import replicate
import requests
from PIL import Image

from . import log, DATA_DIR, OUTPUT_DIR, KEYS_DIR

try:
    with open(os.path.join(KEYS_DIR, 'replicate.txt'), 'r') as f:
        _key = f.read()
        os.environ['REPLICATE_API_TOKEN'] = _key
        REPLICATE_API_TOKEN = _key
except FileNotFoundError:
    log.warning('Replicate API key not found. Some features may not work.')


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
