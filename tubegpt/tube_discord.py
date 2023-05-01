import os
import logging
import discord

log = logging.getLogger(__name__)

def set_discord_key(key: str = None, keys_dir: str = None):
    if key is None:
        try:
            with open(os.path.join(keys_dir, "discord.txt"), "r") as f:
                key = f.read()
        except FileNotFoundError:
            log.warning("Discord API key not found. Some features may not work.")
    os.environ["DISCORD_API_KEY"] = key
    log.info("Discord API key set.")


def send_discord():
    intents = discord.Intents.default()
    intents.message_content = True
    client = MyClient(intents=intents)
    client.run(os.environ["DISCORD_API_KEY"])


class MyClient(discord.Client):
    async def on_ready(self):
        print("Logged on as", self.user)

    async def on_message(self, message):
        # don't respond to ourselves
        if message.author == self.user:
            return

        if message.content == "ping":
            await message.channel.send("pong")


