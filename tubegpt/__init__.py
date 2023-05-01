import logging
import os

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS_DIR = os.path.join(ROOT_DIR, '.keys')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
FONTS_DIR = os.path.join(ROOT_DIR, 'fonts')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')

log.info('Initializing TubeGPT ...')
log.info(f'Looking for keys in: {KEYS_DIR}')
log.info(f'Data will be found in: {DATA_DIR}')
log.info(f'Output will be saved to: {OUTPUT_DIR}')

