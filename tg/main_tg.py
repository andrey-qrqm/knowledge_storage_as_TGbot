import asyncio
from telebot.async_telebot import AsyncTeleBot
import logging
from dotenv import load_dotenv
import os
import time
import json, tempfile, uuid

load_dotenv()
TG_TOKEN = os.getenv("TG_TOKEN")
ADMIN_ID = os.getenv("ADMIN_ID")

PATH_INFO = "data/info.json"
bot = AsyncTeleBot(TG_TOKEN)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

_info_lock = asyncio.Lock()

def check_admin(message) -> bool:
    if message.from_user.id == int(os.getenv("ADMIN_ID")):
        logging.info(f"Admin {message.from_user.username} executed command: {message.text}")
        return True
    logging.warning(f"User {message.from_user.username}, ID: {message.from_user.id} tried: {message.text}")
    return False

def _append_info_sync(data, path=PATH_INFO):
    with open(path, "r", encoding="utf-8") as f:
        current_data = json.load(f)
    current_data.append(data) # append new data to the list

    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(current_data, f, indent=4, ensure_ascii=False)
    os.replace(tmp, path)


async def append_info(data, path=PATH_INFO):
    async with _info_lock:
        await asyncio.to_thread(_append_info_sync, data, path)


@bot.message_handler(commands=['add'])
async def add_text(message):
    if not check_admin(message):
        await bot.reply_to(message, "You are not an admin.")
        return

    parts = message.text.split(maxsplit=1)

    if len(parts) <= 1:
        await bot.reply_to(message, "Usage: /add <info>")
        return

    info = parts[1]
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    data = {
        "index": uuid.uuid4().hex,
        "info": info,
        "timestamp": timestamp
    }
    await append_info(data)
    await bot.reply_to(message, f"Info added: {data['info']}")


async def main():
    await bot.infinity_polling(logger_level=logging.WARNING)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception:
        logging.exception("Bot crashed")
        raise  # non-zero exit, so systemd sees a failure