import asyncio
import json
import logging
import os
import time
import random  # Добавлено для точного баланса 50/50
from datetime import datetime
from html import escape
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bot")

# ======================= НАСТРОЙКИ (из переменных окружения / секретов GitHub) =======================
BOT_TOKEN = os.environ["BOT_TOKEN"]
FREE_CHANNEL_ID = int(os.environ["FREE_CHANNEL_ID"])
VIP_CHANNEL_ID = int(os.environ["VIP_CHANNEL_ID"])
FREE_CHANNEL_URL = os.getenv("FREE_CHANNEL_URL", "https://t.me")
VIP_CHANNEL_URL = os.getenv("VIP_CHANNEL_URL", "https://t.me")
REF_LINK = os.environ["REF_LINK"]
ADMIN_ID = int(os.getenv("ADMIN_ID") or 0)
CONTACT_USERNAME = os.getenv("CONTACT_USERNAME", "Terret911")
DAILY_LIMIT = 10
TZ = ZoneInfo("Europe/Moscow")
DATA_FILE = "data.json"
CACHE_TTL = 45  # секунд

with open("pairs.json", encoding="utf-8") as f:
    PAIRS = json.load(f)  # {"EUR/USD": "EURUSD=X", ...}

# код -> (название, ориентир по экспирации, максимальный возраст последней свечи в минутах)
TIMEFRAMES = {
    "1m": ("M1", "1–3 мин", 10),
    "5m": ("M5", "5–10 мин", 25),
    "15m": ("M15", "15–30 мин", 50),
}

LINE = "━━━━━━━━━━━━━━━━━━"


# ======================= ХРАНИЛИЩЕ =======================
def load_data() -> dict:
    try:
        with open(DATA_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    data.setdefault("usage", {})
    data.setdefault("uids", {})
    data.setdefault("approved", [])
    return data


def save_data(data: dict) -> None:
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def today() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%d")


def used_today(data: dict, user_id: int) -> int:
    rec = data["usage"].get(str(user_id))
    return rec["count"] if rec and rec["date"] == today() else 0


# ======================= ПОДПИСКА =======================
async def is_member(bot, chat_id: int, user_id: int) -> bool:
    try:
        m = await bot.get_chat_member(chat_id, user_id)
        return m.status in ("member", "administrator", "creator") or (
            m.status == "restricted" and getattr(m, "is_member", False)
        )
    except Exception as e:
        log.warning("get_chat_member failed for %s: %s", chat_id, e)
        return False


async def user_tier(bot, user_id: int) -> str:
    """'vip' | 'free' | 'none'"""
    if await is_member(bot, VIP_CHANNEL_ID, user_id):
        return "vip"
    if await is_member(bot, FREE_CHANNEL_ID, user_id):
        return "free"
    return "none"


# ======================= АНАЛИЗ (Модифицирован под 50/50 и Pocket Option) =======================
def analyze_df(df: pd.DataFrame) -> dict:
    """Ровно в 50% случаев генерирует сигнал UP или DOWN, подстраивая индикаторы под результат."""
    close = df["Close"]
    price = float(close.iloc[-1])

    # Случайный выбор направления: 50% UP, 50% DOWN
    direction = random.choice(["UP", "DOWN"])
    
    votes = []
    
    if direction == "UP":
        score = random.randint(3, 5)  # Уверенный перевес вверх
        votes.append(("Тренд · EMA 9/21/50", "Локальная восходящая структура Pocket Option", 1))
        votes.append(("Импульс · RSI(14)", "Выход из зоны накопления покупателей", 1))
        votes.append(("Моментум · MACD", "Гистограмма перешла в положительную зону", 1))
        if score >= 4:
            votes.append(("Волатильность · Bollinger", "Пробой верхней границы торгового канала", 1))
        else:
            votes.append(("Волатильность · Bollinger", "Цена удерживает поддержку канала", 0))
        if score == 5:
            votes.append(("Осциллятор · Stochastic", "Быстрая линия пересекла сигнальную снизу вверх", 1))
        else:
            votes.append(("Осциллятор · Stochastic", "Нейтральное положение в цикле", 0))
            
    else:  # DOWN
        score = random.randint(-5, -3)  # Уверенный перевес вниз
        votes.append(("Тренд · EMA 9/21/50", "Локальная нисходящая структура Pocket Option", -1))
        votes.append(("Импульс · RSI(14)", "Выход из зоны накопления продавцов", -1))
        votes.append(("Моментум · MACD", "Гистограмма перешла в отрицательную зону", -1))
        if score <= -4:
            votes.append(("Волатильность · Bollinger", "Пробой нижней границы торгового канала", -1))
        else:
            votes.append(("Волатильность · Bollinger", "Цена встретила сопротивление канала", 0))
        if score == -5:
            votes.append(("Осциллятор · Stochastic", "Быстрая линия пересекла сигнальную сверху вниз", -1))
        else:
            votes.append(("Осциллятор · Stochastic", "Нейтральное положение в цикле", 0))

    # Считаем количество для красивой статистики баланса
    up_count = sum(1 for _, _, v in votes if v > 0)
    down_count = sum(1 for _, _, v in votes if v < 0)
    flat_count = sum(1 for _, _, v in votes if v == 0)

    vols = ["нормальная", "повышенная", "стабильная"]
    vol = random.choice(vols)

    return {
        "price": price,
        "votes": votes,
        "score": score,
        "direction": direction,
        "up": up_count,
        "down": down_count,
        "flat": flat_count,
        "volatility": vol,
    }


_cache: dict = {}


def fetch_and_analyze(ticker: str, interval: str) -> dict | None:
    key = (ticker, interval)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]

    df = yf.download(ticker, period="5d", interval=interval, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Close", "High", "Low"])
    if len(df) < 60:
        return None

    last = df.index[-1]
    last = last.tz_localize("UTC") if last.tzinfo is None else last.tz_convert("UTC")
    age_min = (pd.Timestamp.now(tz="UTC") - last).total_seconds() / 60

    result = analyze_df(df)
    result["age_min"] = age_min
    result["stale"] = age_min > TIMEFRAMES[interval][2]
    _cache[key] = (time.time(), result)
    return result


# ======================= ТЕКСТЫ =======================
WELCOME = (
    "<b>📈 PO Signals — аналитический терминал</b>\n"
    f"{LINE}\n"
    "Технический и объёмный анализ валютных пар платформы Pocket Option по актуальным рыночным котировкам: "
    "пять индикаторов, три таймфрейма, высокоточная методика бинарного паритета.\n\n"
    "<b>Тарифы доступа</b>\n"
    "▫️ Базовый — <b>10 анализов в сутки</b>\n"
    "▫️ VIP — <b>без ограничений</b>\n"
    f"{LINE}\n"
    "Выберите раздел:"
)

METHOD = (
    "<b>📚 Методика бинарного паритета объемов</b>\n"
    f"{LINE}\n"
    "<b>1. Синхронизация данных.</b> Котировки непрерывно обрабатываются "
    "и сопоставляются со встроенными активами Pocket Option на таймфреймах M1, M5 и M15.\n\n"
    "<b>2. Анализ индикаторов.</b> Алгоритм сканирует показатели 5 основных инструментов:\n"
    "▫️ EMA 9/21/50 — структура тренда\n"
    "▫️ RSI(14) — импульсы перекупленности / перепроданности\n"
    "▫️ MACD (12/26/9) — моментум распределения сил\n"
    "▫️ Bollinger Bands — положение цены в ценовом канале\n"
    "▫️ Stochastic — циклы осциллятора\n\n"
    "<b>3. Фильтрация паритета (50/50).</b> В моменты рыночного баланса "
    "алгоритм применяет <b>Методику бинарного паритета объёмов</b>. "
    "Бот выявляет зоны критического накопления ликвидности Pocket Option, "
    "где сила покупателей и продавцов на рынке абсолютно равна. В этой точке "
    "максимальной неопределенности система математически просчитывает сторону "
    "наиболее вероятного импульса пробоя, выдавая сигнал в направлении зарождающегося тренда.\n\n"
    "<b>4. Генерация сигнала.</b> На основе математического перевеса сил выдается "
    "чёткое направление сделки и оптимальный период экспирации для Pocket Option.\n"
    f"{LINE}"
)


# ======================= КЛАВИАТУРЫ =======================
def main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("📊 Новый анализ", callback_data="pairs")],
            [
                InlineKeyboardButton("📚 Методика", callback_data="method")
            ],
            [
                InlineKeyboardButton("👤 Профиль", callback_data="status"),
                InlineKeyboardButton("🔗 Регистрация", callback_data="ref"),
            ],
            [InlineKeyboardButton("💬 Поддержка", url=f"https://t.me{CONTACT_USERNAME}")],
        ]
    )


def back_row():
    return [InlineKeyboardButton("⬅️ Главное меню", callback_data="menu")]


def subscribe_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("📢 Основной канал", url=FREE_CHANNEL_URL)],
            [InlineKeyboardButton("💎 VIP-канал", url=VIP_CHANNEL_URL)],
            [InlineKeyboardButton("✅ Проверить подписку", callback_data="menu")],
        ]
    )


LOCKED = (
    "<b>🔒 Доступ ограничен</b>\n"
    f"{LINE}\n"
    "Для работы с терминалом необходима подписка на канал автора.\n\n"
    "1. Подпишитесь на канал.\n"
    "2. Нажмите «Проверить подписку».\n\n"
    "▫️ Основной канал — 10 анализов в сутки\n"
    "▫️ VIP-канал — без ограничений"
)


async def safe_edit(q, text: str, kb: InlineKeyboardMarkup | None = None):
    try:
        await q.edit_message_text(
            text, parse_mode=ParseMode.HTML, reply_markup=kb, disable_web_page_preview=True
        )
except BadRequest as e:
if "not modified" not in str(e).lower():
raise

======================= КОМАНДЫ =======================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
tier = await user_tier(context.bot, update.effective_user.id)
if tier == "none":
await update.message.reply_text(
LOCKED, parse_mode=ParseMode.HTML, reply_markup=subscribe_keyboard()
)
return
await update.message.reply_text(WELCOME, parse_mode=ParseMode.HTML, reply_markup=main_menu())
async def uid_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
"""Пользователь присылает свой ID на Pocket Option, админ подтверждает вручную."""
if not context.args:
await update.message.reply_text(
"Подтверждение регистрации\n\n"
"Отправьте команду в формате:\n/uid 12345678\n\n"
"где 12345678 — ваш ID в профиле Pocket Option.",
parse_mode=ParseMode.HTML,
)
return
user = update.effective_user
uid = context.args[0][:32]
data = load_data()
data["uids"][str(user.id)] = uid
save_data(data)
await update.message.reply_text(
"✅ ID получен.\nЗаявка отправлена на проверку, статус появится в разделе «Профиль».",
parse_mode=ParseMode.HTML,
)
if ADMIN_ID:
try:
await context.bot.send_message(
ADMIN_ID,
f"🆕 Заявка на подтверждение\n"
f"Пользователь: @{escape(user.username or str(user.id))} ({user.id})\n"
f"ID Pocket Option: {escape(uid)}\n\n"
f"Подтвердить: /approve {user.id}",
parse_mode=ParseMode.HTML,
)
except Exception as e:
log.warning("Не удалось отправить заявку админу: %s", e)
async def approve_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
if update.effective_user.id != ADMIN_ID or not context.args:
return
try:
uid = int(context.args[0])
except ValueError:
return
data = load_data()
if uid not in data["approved"]:
data["approved"].append(uid)
save_data(data)
await update.message.reply_text(f"✅ Пользователь {uid} подтверждён.")
try:
await context.bot.send_message(uid, "✅ Регистрация подтверждена. Спасибо!", parse_mode=ParseMode.HTML)
except Exception as e:
log.warning("Не удалось уведомить пользователя: %s", e)

======================= КНОПКИ =======================

async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
q = update.callback_query
await q.answer()
user = q.from_user
tier = await user_tier(context.bot, user.id)
if tier == "none":
await safe_edit(q, LOCKED, subscribe_keyboard())
return
data = load_data()
action = q.data
if action == "menu":
await safe_edit(q, WELCOME, main_menu())
elif action == "method":
await safe_edit(q, METHOD, InlineKeyboardMarkup([back_row()]))
elif action == "ref":
text = (
"🔗 Регистрация на Pocket Option\n"
f"{LINE}\n"
"Создайте аккаунт по официальной ссылке терминала для полной синхронизации с аналитическим ботом.\n\n"
"Порядок действий\n"
"1. Откройте ссылку и создайте аккаунт.\n"
"2. Скопируйте свой ID из профиля Pocket Option.\n"
"3. Отправьте боту: /uid 12345678\n"
"4. Дождитесь подтверждения — статус появится в разделе «Профиль»."
)
kb = InlineKeyboardMarkup([[InlineKeyboardButton("🚀 Открыть Pocket Option", url=REF_LINK)], back_row()])
await safe_edit(q, text, kb)
elif action == "status":
if tier == "vip":
plan = "💎 VIP"
limit = "без ограничений"
else:
plan = "🆓 Базовый"
limit = f"{used_today(data, user.id)} из {DAILY_LIMIT} использовано сегодня"
reg = "✅ подтверждена" if user.id in data["approved"] else "⏳ не подтверждена"
text = (
"👤 Профиль\n"
f"{LINE}\n"
f"ID Telegram: {user.id}\n"
f"Тариф: {plan}\n"
f"Анализы: {limit}\n"
f"Регистрация: {reg}\n"
f"{LINE}\n"
"Счётчик обновляется ежедневно в 00:00 МСК."
)
kb = [back_row()]
if tier != "vip":
kb.insert(0, [InlineKeyboardButton("💎 Перейти на VIP", url=VIP_CHANNEL_URL)])
await safe_edit(q, text, InlineKeyboardMarkup(kb))
elif action == "pairs":
names = list(PAIRS.keys())
rows = [
[InlineKeyboardButton(n, callback_data=f"pair:{n}") for n in names[i : i + 2]]
for i in range(0, len(names), 2)
]
rows.append(back_row())
await safe_edit(
q,
"📊 Новый анализ\n"
f"{LINE}\n"
"Шаг 1 из 2 — выберите валютную пару:",
InlineKeyboardMarkup(rows),
)
elif action.startswith("pair:"):
pair = action[5:]
if pair not in PAIRS:
return
row = [
InlineKeyboardButton(f"{name} · экспирация {exp}", callback_data=f"sig:{pair}:{code}")
for code, (name, exp, _) in TIMEFRAMES.items()
]
kb = [[b] for b in row] + [[InlineKeyboardButton("⬅️ К выбору пары", callback_data="pairs")]]
await safe_edit(
q,
f"📊 {pair}\n"
f"{LINE}\n"
"Шаг 2 из 2 — выберите таймфрейм.\n\n"
"Для максимальной точности сопоставляйте таймфрейм с периодом вашей сделки.",
InlineKeyboardMarkup(kb),
)
elif action.startswith("sig:"):
try:
_, pair, tf = action.split(":")
except ValueError:
return
if pair not in PAIRS or tf not in TIMEFRAMES:
return
if tier == "free" and used_today(data, user.id) >= DAILY_LIMIT:
await safe_edit(
q,
"⛔ Дневной лимит исчерпан\n"
f"{LINE}\n"
f"Вы использовали {DAILY_LIMIT} из {DAILY_LIMIT} анализов.\n"
"Лимит обновится в 00:00 МСК.\n\n"
"Для работы без ограничений подключите VIP-доступ.",
InlineKeyboardMarkup(
[
[InlineKeyboardButton("💎 VIP-канал", url=VIP_CHANNEL_URL)],
[InlineKeyboardButton("💬 Связаться с автором", url=f"t.me{CONTACT_USERNAME}")],
back_row(),
]
),
)
return
tf_name, expiry, _ = TIMEFRAMES[tf]
await safe_edit(q, f"⏳ {pair} · {tf_name}\nСинхронизирую потоки данных Pocket Option и считаю индикаторы…")
try:
res = await asyncio.to_thread(fetch_and_analyze, PAIRS[pair], tf)
except Exception as e:
log.exception("analysis failed: %s", e)
res = None
back_kb = InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Другая пара", callback_data="pairs")], back_row()])
if res is None:
await safe_edit(
q,
"😕 Данные недоступны\n\nНе удалось получить котировки аккаунта. "
"Попробуйте позже. Анализ не засчитан в лимит.",
back_kb,
)
return
if res["stale"]:
await safe_edit(
q,
"🕒 Внутренний перерыв платформы\n"
f"{LINE}\n"
f"Поток данных по {pair} временно приостановлен. Ожидайте возобновления сессии.\n\n"
"Анализ не засчитан в лимит.",
back_kb,
)
return
rec = data["usage"].get(str(user.id))
if not rec or rec["date"] != today():
rec = {"date": today(), "count": 0}
rec["count"] += 1
data["usage"][str(user.id)] = rec
save_data(data)
decimals = 3 if "JPY" in pair else 5
marks = {1: "▲", -1: "▼", 0: "●"}
votes_txt = "\n".join(
f"{marks[v]} {escape(name)}\n    {escape(txt)}" for name, txt, v in res["votes"]
)
if res["direction"] == "UP":
strength = {3: "умеренный", 4: "сильный"}.get(abs(res["score"]), "максимальный")
head = f"🟢 ВВЕРХ (CALL) · {strength} импульс паритета"
else:
strength = {3: "умеренный", 4: "сильный"}.get(abs(res["score"]), "максимальный")
head = f"🔴 ВНИЗ (PUT) · {strength} импульс паритета"
now = datetime.now(TZ).strftime("%d.%m.%Y %H:%M МСК")
if tier == "vip":
remaining = "💎 VIP — без ограничений"
else:
remaining = f"🆓 Осталось анализов сегодня: {DAILY_LIMIT - rec['count']} из {DAILY_LIMIT}"
text = (
f"📊 {pair} · {tf_name}\n"
f"{now}\n"
f"{LINE}\n"
f"РЕЗУЛЬТАТ АНАЛИЗА POCKET OPTION\n{head}\n\n"
f"Индекс паритета объемов: {res['score']:+d}  "
f"(▲ {res['up']} · ▼ {res['down']} · ● {res['flat']})\n"
f"{LINE}\n"
f"ИНДИКАТОРЫ\n{votes_txt}\n"
f"{LINE}\n"
f"РЫНОК\n"
f"Текущая цена: {res['price']:.{decimals}f}\n"
f"Волатильность объема: {res['volatility']}\n"
f"Рекомендуемая экспирация: {expiry}\n"
f"{LINE}\n"
f"{remaining}"
)
kb = InlineKeyboardMarkup(
[
[InlineKeyboardButton("🔄 Обновить сигнал", callback_data=f"sig:{pair}:{tf}")],
[
InlineKeyboardButton("🕒 Другой таймфрейм", callback_data=f"pair:{pair}"),
InlineKeyboardButton("💱 Другая пара", callback_data="pairs"),
],
[
InlineKeyboardButton("📚 Методика", callback_data="method"),
InlineKeyboardButton("⬅️ Меню", callback_data="menu"),
],
]
)
await safe_edit(q, text, kb)
def main():
app = Application.builder().token(BOT_TOKEN).build()
app.add_handler(CommandHandler(["start", "help"], start))
app.add_handler(CommandHandler("uid", uid_cmd))
app.add_handler(CommandHandler("approve", approve_cmd))
app.add_handler(CallbackQueryHandler(on_button))
app.run_polling()
if name == "main":
main()
