import os
from io import BytesIO
from typing import Dict, List, Union

import openai
import requests
from PIL import Image

from . import DATA_DIR, OUTPUT_DIR


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
        prompt = [{"role" : "system", "content" : system}] + prompt 
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

