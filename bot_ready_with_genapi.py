
import os
import asyncio
import logging
import base64
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, BufferedInputFile, ReplyKeyboardMarkup, KeyboardButton, BotCommand
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
import aiosqlite
import json
from datetime import date
import requests  # Для Gen-API
from openai import AsyncOpenAI  # Для Aitunnel

# Конфиг
TELEGRAM_TOKEN = "8217361037AAEgJ6NugPqXDNXstIOL5g7R1ovBxsLAWM"
ADMIN_ID = 6387718314
AITUNNEL_KEY = "sk-aitunnel-9ho4TkDH1Vxr0koqvpQtPS1mL2Yyv1v8"  # Твой Aitunnel ключ
GEN_API_KEY = "sk-dd7I7EH6Gtg0zBTDManlSPCLoBN8rQPAatfF57GFebec8vgBHVbnx15JTKMa"  # Gen-API ключ
DB_FILE = "data/bot_data.db"
FREE_DAILY_LIMIT = 20
PREMIUM_DAILY_LIMIT = 200

logging.basicConfig(level=logging.INFO)
bot = Bot(token=TELEGRAM_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

FREE_MODELS = ["gpt-4o-mini"]
VISION_MODELS = ["gpt-4o", "gpt-4o-mini"]

class ImageGenState(StatesGroup):
    waiting_for_prompt = State()

class BotDatabase:
    # Твой оригинальный код БД (сокращён для примера)
    def __init__(self): self.db_path = DB_FILE
    async def init_db(self): pass  # Вставь полный init_db из file:1
    async def get_user(self, user_id): return {"is_premium": False, "current_model": "gpt-4o-mini"}
    async def check_limit(self, user_id): return True, 10
    async def increment_messages(self, user_id): pass

db = BotDatabase()
aitunnel_client = AsyncOpenAI(api_key=AITUNNEL_KEY, base_url="https://api.aitunnel.ru/v1/")

async def get_ai_response(user_message: str, model: str = "gpt-4o-mini", image_b64: str = None):
    messages = [{"role": "user", "content": user_message}]
    if image_b64 and model in VISION_MODELS:
        messages[0]["content"] = [
            {"type": "text", "text": user_message},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
        ]
    try:
        response = await aitunnel_client.chat.completions.create(
            model=model, messages=messages, max_tokens=4000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Ошибка API: {e}"

async def generate_image_genapi(prompt: str, image_b64: str = None) -> bytes:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GEN_API_KEY}"}
    data = {"callback_url": None, "prompt": prompt}
    if image_b64: data["image"] = image_b64
    resp = requests.post("https://api.gen-api.ru/api/v1/networks/seededit", json=data, headers=headers)
    if resp.status_code != 200: return None
    req_id = resp.json().get("request_id")
    for _ in range(24):
        await asyncio.sleep(5)
        poll = requests.get(f"https://api.gen-api.ru/api/v1/requests/{req_id}", headers=headers)
        if poll.json().get("status") == "success":
            img_url = poll.json()["output"]["url"]
            return requests.get(img_url).content
    return None

def get_keyboard():
    return ReplyKeyboardMarkup(keyboard=[
        [KeyboardButton(text="🖼️ Генерация"), KeyboardButton(text="📊 Статус")]
    ], resize_keyboard=True)

@dp.message(Command("start"))
async def start(message: Message):
    await message.answer("🚀 Бот готов! Отправь текст/фото.", reply_markup=get_keyboard())

@dp.message(F.text)
async def text_msg(message: Message):
    response = await get_ai_response(message.text)
    await message.answer(response)

@dp.message(F.photo)
async def photo_msg(message: Message, state: FSMContext):
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    photo_bytes = await bot.download_file(file.file_path)
    photo_b64 = base64.b64encode(photo_bytes.getvalue()).decode()
    await state.update_data(photo_b64=photo_b64)

    if message.caption:
        # Vision анализ
        resp = await get_ai_response(message.caption, "gpt-4o", photo_b64)
        await message.answer(resp)
    else:
        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton("✨ Редактировать", callback_data="edit_photo")]
        ])
        await message.answer("Выбери:", reply_markup=kb)

@dp.message(Command("genimg"))
async def genimg(message: Message, state: FSMContext):
    await message.answer("Промпт для генерации фото:")
    await ImageGenState.waiting_for_prompt.set()

@dp.message(ImageGenState.waiting_for_prompt)
async def gen_prompt(message: Message, state: FSMContext):
    data = await state.get_data()
    photo_b64 = data.get('photo_b64')
    img_data = await generate_image_genapi(message.text, photo_b64)
    if img_data:
        await message.answer_photo(BufferedInputFile(img_data, "generated.png"))
    else:
        await message.answer("❌ Ошибка Gen-API (проверь ключ)")
    await state.clear()

async def main():
    await db.init_db()
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
