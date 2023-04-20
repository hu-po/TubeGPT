import random
import os
import uuid

from src import (OUTPUT_DIR, COLORS, draw_text, gpt_image, gpt_text, remove_bg,
                 stack_fgbg)

for _ in range(4):

    _system = \
        "You generate variations of string prompts for image generation." + \
        "Respond with a single new variant of the user prompt. " + \
        "Add several new interesting or related words." + \
        "Respond with the prompt only: no extra text or explanations."
    fg_prompt = gpt_text(
        prompt="white bengal cat, blue eyes, cute, chubby",
        system=_system,
    )
    bg_prompt = gpt_text(
        prompt="space background, digital universe, galaxies",
        system=_system,
    )

    # Foreground image remove background
    fg_img_name = str(uuid.uuid4())
    gpt_image(
        prompt=fg_prompt,
        output_path=os.path.join(OUTPUT_DIR, f'{fg_img_name}.png'),
        image_size="512x512",
    )
    remove_bg(
        image_path=os.path.join(OUTPUT_DIR, f'{fg_img_name}.png'),
        output_path=os.path.join(OUTPUT_DIR, f'{fg_img_name}_nobg.png'),
    )

    # Background image 
    bg_img_name = str(uuid.uuid4())
    gpt_image(
        prompt=bg_prompt,
        output_path=os.path.join(OUTPUT_DIR, f'{bg_img_name}.png'),
        # TODO: Background needs to be 16:9 aspect ratio
        # OpenAI only supports 1:1
        image_size="512x512",
    )

    for _ in range(8):

        combo_image_name = str(uuid.uuid4())

        # Stack foreground and background
        stack_fgbg(
            fg_image_path = os.path.join(OUTPUT_DIR, f'{fg_img_name}_nobg.png'),
            bg_image_path = os.path.join(OUTPUT_DIR, f'{bg_img_name}.png'),
            output_path = os.path.join(OUTPUT_DIR, f'{combo_image_name}.png'),
            bg_size = (1280, 720),
            fg_size = (420, 420),
            gaussian_mu_sig = ((0.05, 0.05), (0.5, 0.05)),
        )

        # Write text on top of image
        _available_colors = list(COLORS.keys())
        draw_text(
            image_path = os.path.join(OUTPUT_DIR, f'{combo_image_name}.png'),
            output_path = os.path.join(OUTPUT_DIR,  f'{combo_image_name}_text.png'),
            text = 'AI Generated YouTube Thumbnails',
            text_color = random.choice(_available_colors), 
            font = 'hupo',
            font_size = random.choice([50, 70, 100]),
            rectangle_color = random.choice(_available_colors),
            rectangle_padding = random.choice([10, 20, 40,]),
        )