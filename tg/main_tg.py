import asyncio
from telebot.async_telebot import AsyncTeleBot
import logging
from dotenv import load_dotenv
import os
import time
import json, tempfile, uuid

from retrieve.retriever import retrieve_context, build_context
import httpx

load_dotenv()
TG_TOKEN = os.getenv("TG_TOKEN")
ADMIN_ID = os.getenv("ADMIN_ID")
PROXY_URL = os.getenv("PROXY_URL")

PATH_INFO = "tg/data/info.json"
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
        "timestamp": timestamp,
        "author": message.from_user.username,
        "user_id": message.from_user.id,
        "is_embedded": False
    }
    await append_info(data)
    await bot.reply_to(message, f"Info added: {data['info']}")

@bot.message_handler(commands=['ask'])
async def ask_question(message):
    if not check_admin(message):
        await bot.reply_to(message, "You are not an admin.")
        return

    parts = message.text.split(maxsplit=1)

    if len(parts) <= 1:
        await bot.reply_to(message, "Usage: /ask <question>")
        return

    question = parts[1]
    logging.info(f"Received question: {question}")

    data = {
        "model": "llama3.2:3b",
        "stream": False,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ]
    }
    # Forward to proxy
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(PROXY_URL, json=data)
            response.raise_for_status()
            response_data = response.json()
            answer = response_data["message"]["content"].strip()
            logging.info(f"Received answer: {answer}")
        except httpx.TimeoutException:
            logging.error("Request to proxy timed out.")
            answer = "Error: The request to the proxy service timed out."
        except httpx.RequestError as e:
            logging.error(f"Request error while forwarding question: {e}")
            answer = "Error: Could not reach the proxy service."
        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP error while forwarding question: {e}")
            answer = f"Error: Received bad response from proxy service: {e.response.status_code}"
        except Exception as e:
            logging.exception("Unexpected error while forwarding question")
            answer = "Error: An unexpected error occurred."
        
        await bot.reply_to(message, answer)


@bot.message_handler(commands=['ask_context'])
async def ask_question_with_context(message):
    if not check_admin(message):
        await bot.reply_to(message, "You are not an admin.")
        return

    parts = message.text.split(maxsplit=1)

    if len(parts) <= 1:
        await bot.reply_to(message, "Usage: /ask_context <question>")
        return

    question = parts[1]
    logging.info(f"Received question for context retrieval: {question}")

    # Retrieve context
    results = retrieve_context(question, top_k=5)
    context = build_context(results, max_chars=6000)
    logging.info(f"Built context for question: {len(context)} characters")

    # Prepare the prompt with context
    prompt = f"Context:\n{context}\n\nQuestion: {question}"

    data = {
        "model": "llama3.2:3b",
        "stream": False,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }

    # Forward to proxy
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(PROXY_URL, json=data)
            response.raise_for_status()
            response_data = response.json()
            answer = response_data["message"]["content"].strip()
            logging.info(f"Received answer with context: {answer}")
        except httpx.TimeoutException:
            logging.error("Request to proxy timed out.")
            answer = "Error: The request to the proxy service timed out."
        except httpx.RequestError as e:
            logging.error(f"Request error while forwarding question with context: {e}")
            answer = "Error: Could not reach the proxy service."
        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP error while forwarding question with context: {e}")
            answer = f"Error: Received bad response from proxy service: {e.response.status_code}"
        except Exception as e:
            logging.exception("Unexpected error while forwarding question with context")
            answer = "Error: An unexpected error occurred."
        
        await bot.reply_to(message, answer)

async def main():
    await bot.infinity_polling(logger_level=logging.DEBUG)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception:
        logging.exception("Bot crashed")
        raise  # non-zero exit, so systemd sees a failure