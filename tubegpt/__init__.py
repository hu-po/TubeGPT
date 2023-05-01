import logging
log = logging.getLogger(__name__)
try:
    from .tube_arxiv import *
except ImportError:
    log.warning("ArXiV not imported, perhaps try (pip install arxiv).")
try:
    from .tube_discord import *
except ImportError:
    log.warning("Discord not imported, perhaps try (pip install discord).")
try:
    from .tube_github import *
except ImportError:
    log.warning("GitHub not imported, perhaps try (pip install PyGithub).")
try:
    from .tube_pillow import *
except ImportError:
    log.warning("Pillow not imported, perhaps try (pip install Pillow).")
try:
    from .tube_notion import *
except ImportError:
    log.warning("Notion not imported, perhaps try (pip install notion).")
try:
    from .tube_openai import *
except ImportError:
    log.warning("OpenAI not imported, perhaps try (pip install openai).")
try:
    from .tube_replicate import *
except ImportError:
    log.warning("Replicate not imported, perhaps try (pip install replicate).")
try:
    from .tube_google import *
except ImportError:
    log.warning("Google not imported, perhaps try (pip install google).")
try:
    from .tube_elevenlabs import *
except ImportError:
    log.warning("ElevenLabs not imported, perhaps try (pip install elevenlabs).")
