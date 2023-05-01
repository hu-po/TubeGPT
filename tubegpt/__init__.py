import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())
__all__ = []
try:
    import tube_arxiv as tube_arxiv
    __all__.append("tube_arxiv")
except ImportError:
    log.warning("ArXiV not imported, perhaps try (pip install arxiv).")
try:
    from .tube_discord import *

    __all__.append("tube_discord")
except ImportError:
    log.warning("Discord not imported, perhaps try (pip install discord).")
try:
    from .tube_github import *

    __all__.append("tube_github")
except ImportError:
    log.warning("GitHub not imported, perhaps try (pip install PyGithub).")
try:
    from .tube_pillow import *

    __all__.append("tube_pillow")
except ImportError:
    log.warning("Pillow not imported, perhaps try (pip install Pillow).")
try:
    from .tube_notion import *

    __all__.append("tube_notion")
except ImportError:
    log.warning("Notion not imported, perhaps try (pip install notion).")
try:
    from .tube_openai import *

    __all__.append("tube_openai")
except ImportError:
    log.warning("OpenAI not imported, perhaps try (pip install openai).")
try:
    from .tube_replicate import *

    __all__.append("tube_replicate")
except ImportError:
    log.warning("Replicate not imported, perhaps try (pip install replicate).")
try:
    from .tube_google import *

    __all__.append("tube_google")
except ImportError:
    log.warning("Google not imported, perhaps try (pip install google).")
try:
    from .tube_elevenlabs import *

    __all__.append("tube_elevenlabs")
except ImportError:
    log.warning("ElevenLabs not imported, perhaps try (pip install elevenlabs).")
