import random
import os
import uuid

from ops import (OUTPUT_DIR, DATA_DIR, draw_text, gpt_image, gpt_text, remove_bg,
                 stack_fgbg, gpt_color, resize_bg)

for _ in range(2):

    fg_prompt = gpt_text(
        prompt="portrait of white bengal cat, blue eyes, cute, chubby",
        system=' '.join([
        "You generate variations of string prompts for image generation.",
        "Respond with a single new variant of the user prompt. ",
        "Add several new interesting or related words.",
        "Respond with the prompt only: no extra text or explanations.",
    ]),
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
    resize_bg(
        image_path = os.path.join(DATA_DIR, 'example_graphs.png'),
        output_path = os.path.join(OUTPUT_DIR, f'{bg_img_name}.png'),
        canvas_size = (1280, 720),
    )
    for _ in range(4):

        text_rgb, text_color = gpt_color()
        rect_rgb, rect_color = gpt_color()

        # Write text on top of image
        draw_text(
            image_path = os.path.join(OUTPUT_DIR, f'{bg_img_name}.png'),
            output_path = os.path.join(OUTPUT_DIR,  f'{bg_img_name}_text.png'),
            text = f'{text_color} - {rect_color}',
            text_color = text_rgb,
            font_size = random.choice([50, 70]),
            rectangle_color = rect_rgb,
            rectangle_padding = random.choice([20, 40]),
        )

        # Stack foreground and background
        combo_image_name = str(uuid.uuid4())
        stack_fgbg(
            fg_image_path = os.path.join(OUTPUT_DIR, f'{fg_img_name}_nobg.png'),
            bg_image_path = os.path.join(OUTPUT_DIR, f'{bg_img_name}_text.png'),
            output_path = os.path.join(OUTPUT_DIR, f'_{combo_image_name}.png'),
            bg_size = (1280, 720),
            fg_size = (420, 420),
            gaussian_mu_sig = ((0.05, 0.05), (0.5, 0.05)),
        )

