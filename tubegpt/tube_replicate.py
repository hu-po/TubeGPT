import os
from io import BytesIO
import logging
import replicate
import requests
from PIL import Image

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
