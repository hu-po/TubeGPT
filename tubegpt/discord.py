import os

from . import KEYS_DIR, log

try:
    import discord

    with open(os.path.join(KEYS_DIR, "discord.txt"), "r") as f:
        _key = f.read()
        os.environ["DISCORD_API_KEY"] = _key
        DISCORD_API_KEY = _key
except ImportError:
    log.warning("Discord API not installed (pip install discord.py)")
except FileNotFoundError:
    log.warning("Discord API key not found. Some features may not work.")


class MyClient(discord.Client):
    async def on_ready(self):
        print("Logged on as", self.user)

    async def on_message(self, message):
        # don't respond to ourselves
        if message.author == self.user:
            return

        if message.content == "ping":
            await message.channel.send("pong")


intents = discord.Intents.default()
intents.message_content = True
client = MyClient(intents=intents)
client.run(DISCORD_API_KEY)
