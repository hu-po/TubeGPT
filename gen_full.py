import shutil
import os

from ops import (OUTPUT_DIR, DATA_DIR, generate_yttext, generate_thumbnails)

desired_sentence = '''In this stream we review the paper: "A Method for Animating Children's Drawings of the Human Figure" out of Meta AI Research.

https://paperswithcode.com/paper/a-method-for-automatically-animating-children
https://github.com/facebookresearch/AnimatedDrawings
'''
output_dir_name = 'animated_drawings'
background_image = os.path.join(DATA_DIR, '2023-04-21_10-30.png')

# Create output dir if it doesn't exist
output_dir = os.path.join(OUTPUT_DIR, output_dir_name)
output_tmp_dir = os.path.join(OUTPUT_DIR, output_dir_name, 'tmp')
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
if os.path.exists(output_tmp_dir):
    shutil.rmtree(output_tmp_dir)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_tmp_dir, exist_ok=True)

title = generate_yttext(
    output_dir,
    desired_sentence,
    [
        "eMFfMz9uYlc",
        "vzGy6Att_Hk",
        "RLcr4bqGsEQ",
        "KSZiJ4k28b4",
        "8b6NhnNYtpg",
    ],
)
generate_thumbnails(
    background_image,
    output_tmp_dir,
    output_dir,
    title=title,
)