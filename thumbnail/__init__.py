import logging
import os

from ops_img import *

log = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
FONTS_DIR = os.path.join(ROOT_DIR, 'fonts')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')

try:
    import openai
    with open(os.path.join(ROOT_DIR, 'openai.txt'), 'r') as f:
        _key = f.read()
        os.environ['OPENAI_API_KEY'] = _key
        openai.api_key = _key
        OPENAI_API_KEY = _key
except ImportError:
    log.warning('OpenAI API not installed (pip install openai)')
except FileNotFoundError:
    log.warning('OpenAI API key not found. Some features may not work.')
    
try:
    from ops_arxiv import *
except ImportError:
    log.warning('Arxiv API not installed (pip install arxiv)')

try:
    from ops_google import *
    with open(os.path.join(ROOT_DIR, 'google.txt'), 'r') as f:
        _key = f.read()
        os.environ["GOOGLE_API_KEY"] = _key
        GOOGLE_API_KEY = _key
except ImportError:
    log.warning('Google API not installed (pip install google-api-python-client)')
except FileNotFoundError:
    log.warning('Google API key not found. Some features may not work.')

try:
    from ops_replicate import *
    with open(os.path.join(ROOT_DIR, 'replicate.txt'), 'r') as f:
        _key = f.read()
        os.environ['REPLICATE_API_TOKEN'] = _key
        REPLICATE_API_TOKEN = _key
except ImportError:
    log.warning('Replicate API not installed (pip install replicate)')
except FileNotFoundError:
    log.warning('Replicate API key not found. Some features may not work.')

try:
    from ops_notion import *
    with open(os.path.join(ROOT_DIR, 'notion.txt'), 'r') as f:
        _key = f.read()
        os.environ['NOTION_API_KEY'] = _key
        NOTION_API_KEY = _key
except ImportError:
    log.warning('Notion API not installed (pip install notion-client)')
except FileNotFoundError:
    log.warning('Notion API key not found. Some features may not work.')
