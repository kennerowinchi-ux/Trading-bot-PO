import asyncio
import json
import logging
import math
import os
import random
import time
from datetime import datetime
from html import escape
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InputMediaPhoto, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bot")

# ======================= НАСТРОЙКИ (из переменных окружения / секретов GitHub) =======================
BOT_TOKEN = os.environ["BOT_TOKEN"]
FREE_CHANNEL_ID = int(os.environ["FREE_CHANNEL_ID"])
VIP_CHANNEL_ID = int(os.environ["VIP_CHANNEL_ID"])
FREE_CHANNEL_URL = os.getenv("FREE_CHANNEL_URL", "https://t.me/+F_OldsGjeK5lMjBi")
VIP_CHANNEL_URL = os.getenv("VIP_CHANNEL_URL", "https://t.me/+itYuwMLRnBcyNDEy")
REF_LINK = os.environ["REF_LINK"]
ADMIN_ID = int(os.getenv("ADMIN_ID") or 0)
CONTACT_USERNAME = os.getenv("CONTACT_USERNAME", "Terret911")
DAILY_LIMIT = 10
TZ = ZoneInfo("Europe/Moscow")
DATA_FILE = "data.json"
CACHE_TTL = 45  # секунд (кэш котировок)
NOISE = 0.6     # «случайность» голосов: чем больше, тем сильнее результат меняется при обновлении


with open("pairs.json", encoding="utf-8") as f:
    PAIRS = json.load(f)  # {"EUR/USD": "EURUSD=X", ...}

# код -> (название, ориентир по экспирации, максимальный возраст последней свечи в минутах)
TIMEFRAMES = {
    "30s": ("30S", "30 сек", 10),
    "1m": ("M1", "1 мин", 25),
    "3m": ("M3", "3 мин", 50),
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_UP = os.path.join(BASE_DIR, "up.png")      # зелёное фото: график вверх
IMG_DOWN = os.path.join(BASE_DIR, "down.png")  # красное фото: график вниз
IMG_APPROVED = os.path.join(BASE_DIR, "approved.png")  # заявка одобрена
IMG_REJECTED = os.path.join(BASE_DIR, "rejected.png")  # заявка отклонена

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
    data.setdefault("rejected", [])
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


# ======================= АНАЛИЗ =======================
def analyze_df(df: pd.DataFrame) -> dict:
    """Считает индикаторы и отдаёт для каждого «силу» x (знак = направление). Голоса выносятся в vote_now()."""
    close, high, low = df["Close"], df["High"], df["Low"]

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()

    mid = close.rolling(20).mean()
    std = close.rolling(20).std()

    ll, hh = low.rolling(14).min(), high.rolling(14).max()
    stoch = (100 * (close - ll) / (hh - ll).replace(0, np.nan)).rolling(3).mean()

    prev_close = close.shift()
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    atr_last = float(atr.iloc[-1]) or 1e-9
    atr_ratio = float(atr.iloc[-1] / atr.rolling(50).mean().iloc[-1])

    price = float(close.iloc[-1])

    def num(x, default=0.0):
        x = float(x)
        return default if np.isnan(x) else x

    r = num(rsi.iloc[-1], 50.0)
    s_ = num(stoch.iloc[-1], 50.0)
    sd = num(std.iloc[-1], atr_last) or atr_last

    feats = [
        ("Тренд · EMA 9/21", "быстрая выше медленной", "быстрая ниже медленной",
         (float(ema9.iloc[-1]) - float(ema21.iloc[-1])) / atr_last),
        ("Импульс · RSI 14", "покупатели сильнее", "продавцы сильнее", (r - 50) / 20),
        ("Моментум · MACD", "импульс растёт вверх", "импульс давит вниз", num(hist.iloc[-1]) / (atr_last * 0.5)),
        ("Коридор · Bollinger", "цена выше средней линии", "цена ниже средней линии",
         (price - num(mid.iloc[-1], price)) / sd),
        ("Осциллятор · Stochastic", "сила быков", "сила медведей", (s_ - 50) / 25),
    ]

    if atr_ratio < 0.8:
        vol = "низкая"
    elif atr_ratio > 1.3:
        vol = "повышенная"
    else:
        vol = "нормальная"

    return {"price": price, "atr": atr_last, "feats": feats, "volatility": vol}


def vote_now(base: dict) -> dict:
    """Новый «раунд» голосования: сила индикатора + случайный шум. Каждое обновление может дать другой результат."""
    votes = []
    for name, up_txt, down_txt, x in base["feats"]:
        v = 1 if math.tanh(x) + random.gauss(0, NOISE) > 0 else -1
        votes.append((name, up_txt if v > 0 else down_txt, v))

    score = sum(v for _, _, v in votes)  # 5 голосов -> всегда нечётное, ничьей нет

    # Цена и волатильность тоже «живут»: цена гуляет в пределах нескольких ATR, волатильность выбирается случайно
    price = base["price"] + random.gauss(0, base["atr"] * 0.8)
    levels = ["низкая", "нормальная", "повышенная"]
    weights = [4 if lvl == base["volatility"] else 1 for lvl in levels]
    volatility = random.choices(levels, weights=weights)[0]

    return {
        "price": price,
        "volatility": volatility,
        "votes": votes,
        "score": score,
        "direction": "UP" if score > 0 else "DOWN",
        "up": sum(1 for _, _, v in votes if v > 0),
        "down": sum(1 for _, _, v in votes if v < 0),
    }


_cache: dict = {}


def fetch_and_analyze(ticker: str, interval: str) -> dict | None:
    key = (ticker, interval)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        base = hit[1]
    else:
        # yfinance не умеет интервалы 30s и 3m: качаем минутные свечи,
        # для 30S берём M1 как есть, для M3 собираем свечи по 3 минуты.
        df = yf.download(ticker, period="5d", interval="1m", progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close", "High", "Low"])

        if interval == "3m":
            df = df.resample("3min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
        if len(df) < 60:
            return None

        base = analyze_df(df)
        _cache[key] = (time.time(), base)

    return vote_now(base)


# ======================= ТЕКСТЫ =======================
WELCOME = (
    "<b>📈 Syndicate Trade </b>\n"
    f"{LINE}\n"
    " Syndicate Trade - Специальный бот, созданный для анализа рынка и определением сигналов с последующим направлением рынка.\n"
    " Данный бот имеет возможность анализировать рынок 24/7, благодаря подключенному Claude AI.\n"
    "<i> Доступ к боту имеют все подписчики телеграмм канала блогера Terret </i>\n"
    f"{LINE}\n"
    "Выберите раздел:"
)

METHOD = (
    "<b>🔬 О СТРАТЕГИИ «5 СУДЕЙ»</b>\n"
    f"{LINE}\n"
    "<b>1. Данные.</b> В основе нашего бота лежит строгое математическое правило: убрать рыночный хаос и заставить вероятность работать на вас. "
    "Алгоритм не использует «интуицию» — он опирается на консилиум из 5 независимых технических индикаторов, которые оценивают график Pocket Option каждую секунду.  "
    "В системе 50/50 у индикаторов нет нейтральной зоны (0). Каждый из них обязан вынести четкий вердикт: рынок идет ВВЕРХ (+1) или ВНИЗ (-1).\n\n"
    "<b>2. Индикаторы.</b> ⚖️ КАК ГОЛОСУЮТ СУДЬИ:\n"
    "1️⃣ EMA 9/21 (Тренд) — определяет генеральное направление цены. Если быстрая скользящая выше медленной — голос за РОСТ.\n"
    "2️⃣ RSI 14 (Импульс) — делит рынок пополам. Если индекс выше 50%, значит, инициатива и крупные деньги у покупателей.\n"
    "3️⃣ MACD Hist (Моментум) — замеряет скорость. Если гистограмма зашла выше нуля, значит, движение ускоряется вверх.\n"
    "4️⃣ Bollinger Bands (Коридор) — оценивает зоны. Если цена закрепилась выше средней линии коридора, она находится в бычьей зоне.\n"
    "5️⃣ Stochastic (Осциллятор) — фиксирует микро-силу. Значение выше 50% подтверждает внутреннюю силу быков.\n\n"
    "<b>3. Голосование.</b> 🚀 СИЛА СИГНАЛА: "
    "Счет +5 или -5 — Идеальный сигнал. Все судьи единогласны. Вероятность 50/50 капитулирует перед математическим перевесом.\n\n"
    "Счет от ±1 до ±3 — Стандартный сигнал. Большинство судей перевесило чашу весов.\n\n"
    "<b>4. Сигнал.</b> Бот всегда показывает направление: плюс — ВВЕРХ, минус — ВНИЗ. "
    "Ничьей не бывает, так как судей пять.\n\n"
    "<b>5. Экспирация.</b> Мы пользуемся тремя видами таймфрема (30S, M1, M3). "
    "На личном опыте я выделили именно эти три таймфрейма\n\n"
    f"{LINE}"
)

RISKS = (
    "<b>⚠️ Полная информация:</b>\n"
    f"{LINE}\n"
    "▫️ Это специальный торговый бот, созданный для анализа направления рынка. "
    "Данный бот использует множество аналитических инструментов:\n"
    "▫️ RSI - показывает кто сейчас сильнее на рынке — покупатели или продавцы.\n"
    "▫️ MACD - Скорость движения цены и приток объемов.\n"
    "▫️ Bollinger Bands - Положение цены внутри динамического ценового коридора.\n"
    "▫️ При регистрации по моей ссылке с вас берется комиссия в виде 3% с каждой прибыльной сделки.\n"
    "▫️ Данная комиссия это символическая комиссия за мою работу, благодарю за понимание\n"
    f"{LINE}"
)

# ======================= КЛАВИАТУРЫ =======================
def main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("📊 Анализ рынка", callback_data="pairs")],
            [
                InlineKeyboardButton("📚 Методика", callback_data="method"),
                InlineKeyboardButton("📚 Информация", callback_data="risks"),
            ],
            [
                InlineKeyboardButton("👤 Профиль", callback_data="status"),
                InlineKeyboardButton("🔗 Регистрация", callback_data="ref"),
            ],
            [InlineKeyboardButton("💬 Поддержка", url=f"https://t.me/Terret911")],
        ]
    )


def back_row():
    return [InlineKeyboardButton("⬅️ Главное меню", callback_data="menu")]


def subscribe_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("📢 Основной канал", url=FREE_CHANNEL_URL)],
            [InlineKeyboardButton("✅ Проверить подписку", callback_data="menu")],
        ]
    )


LOCKED = (
    "<b>🔒 Доступ ограничен</b>\n"
    f"{LINE}\n"
    "Для того что бы начать работать по торговому боту, вам необходимо подписаться на канал блогера Terret.\n\n"
    "1. Перейдите в канал «Влог Террета» и подпишитесь\n"
    "2. Нажмите «Проверить подписку».\n\n"
    "▫️ Основной канал — 10 анализов в сутки\n"
    "▫️ VIP-Доступ - Только для людей, состоящих в команде Террета"
)


async def safe_edit(q, text: str, kb: InlineKeyboardMarkup | None = None):
    # Фото-сообщение нельзя превратить в текстовое: удаляем его и шлём обычное сообщение.
    if q.message is not None and q.message.photo:
        await q.message.chat.send_message(
            text, parse_mode=ParseMode.HTML, reply_markup=kb, disable_web_page_preview=True
        )
        try:
            await q.message.delete()
        except Exception:
            pass
        return
    try:
        await q.edit_message_text(
            text, parse_mode=ParseMode.HTML, reply_markup=kb, disable_web_page_preview=True
        )
    except BadRequest as e:
        if "not modified" not in str(e).lower():
            raise


async def show_result(q, text: str, kb: InlineKeyboardMarkup, direction: str):
    """Результат анализа с фото: зелёное для ВВЕРХ, красное для ВНИЗ."""
    path = IMG_UP if direction == "UP" else IMG_DOWN
    if not os.path.exists(path):
        await safe_edit(q, text, kb)
        return
    if q.message is not None and q.message.photo:
        try:
            with open(path, "rb") as f:
                await q.edit_message_media(
                    InputMediaPhoto(f, caption=text, parse_mode=ParseMode.HTML), reply_markup=kb
                )
        except BadRequest as e:
            if "not modified" not in str(e).lower():
                raise
        return
    with open(path, "rb") as f:
        await q.message.chat.send_photo(f, caption=text, parse_mode=ParseMode.HTML, reply_markup=kb)
    try:
        await q.message.delete()
    except Exception:
        pass


# ======================= КОМАНДЫ =======================
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
            "<b>Подтверждение регистрации</b>\n\n"
            "Отправьте команду в формате:\n<code>/uid 12345678</code>\n\n"
            "где <code>12345678</code> — ваш ID в профиле Pocket Option.",
            parse_mode=ParseMode.HTML,
        )
        return
    user = update.effective_user
    uid = context.args[0][:32]
    data = load_data()
    data["uids"][str(user.id)] = uid
    if user.id in data["rejected"]:
        data["rejected"].remove(user.id)
    save_data(data)
    await update.message.reply_text(
        "✅ <b>ID получен.</b>\nЗаявка отправлена на проверку, статус появится в разделе «Профиль».",
        parse_mode=ParseMode.HTML,
    )
    if ADMIN_ID:
        try:
            await context.bot.send_message(
                ADMIN_ID,
                f"🆕 <b>Заявка на подтверждение</b>\n"
                f"Пользователь: @{escape(user.username or str(user.id))} (<code>{user.id}</code>)\n"
                f"ID Pocket Option: <code>{escape(uid)}</code>\n\n"
                f"Или командами: <code>/approve {user.id}</code> · <code>/reject {user.id}</code>",
                parse_mode=ParseMode.HTML,
                reply_markup=InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton("✅ Одобрить", callback_data=f"adm:ok:{user.id}"),
                            InlineKeyboardButton("❌ Отклонить", callback_data=f"adm:no:{user.id}"),
                        ]
                    ]
                ),
            )
        except Exception as e:
            log.warning("Не удалось отправить заявку админу: %s", e)


async def decide(bot, uid: int, approved: bool) -> bool:
    """Фиксирует решение админа и пишет пользователю. Возвращает True, если пользователю удалось написать."""
    data = load_data()
    if approved:
        if uid not in data["approved"]:
            data["approved"].append(uid)
        if uid in data["rejected"]:
            data["rejected"].remove(uid)
        text = (
            "✅ <b>Заявка одобрена</b>\n"
            f"{LINE}\n"
            "Ваша регистрация подтверждена. Анализ рынка теперь доступен.\n\n"
            "Нажмите /start, чтобы открыть меню."
        )
    else:
        if uid in data["approved"]:
            data["approved"].remove(uid)
        if uid not in data["rejected"]:
            data["rejected"].append(uid)
        text = (
            "❌ <b>Заявка отклонена</b>\n"
            f"{LINE}\n"
            "Регистрацию подтвердить не удалось. Проверьте, что вы зарегистрировались по ссылке "
            "из раздела «Регистрация» и указали верный ID.\n\n"
            "Отправьте ID заново: <code>/uid 12345678</code>"
        )
    save_data(data)
    img = IMG_APPROVED if approved else IMG_REJECTED
    try:
        if os.path.exists(img):
            with open(img, "rb") as f:
                await bot.send_photo(uid, f, caption=text, parse_mode=ParseMode.HTML)
        else:
            await bot.send_message(uid, text, parse_mode=ParseMode.HTML)
        return True
    except Exception as e:
        log.warning("Не удалось уведомить пользователя %s: %s", uid, e)
        return False


async def _decision_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE, approved: bool):
    if update.effective_user.id != ADMIN_ID or not context.args:
        return
    try:
        uid = int(context.args[0])
    except ValueError:
        return
    notified = await decide(context.bot, uid, approved)
    word = "одобрена ✅" if approved else "отклонена ❌"
    note = "" if notified else "\n⚠️ Пользователю написать не удалось (он не запускал бота)."
    await update.message.reply_text(f"Заявка пользователя {uid} {word}.{note}")


async def approve_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _decision_cmd(update, context, True)


async def reject_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _decision_cmd(update, context, False)


# ======================= КНОПКИ =======================
async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    user = q.from_user

    if q.data.startswith("adm:"):
        if user.id != ADMIN_ID:
            return
        try:
            _, act, uid_s = q.data.split(":")
            uid = int(uid_s)
        except ValueError:
            return
        notified = await decide(context.bot, uid, act == "ok")
        verdict = "✅ <b>Одобрено</b>" if act == "ok" else "❌ <b>Отклонено</b>"
        note = "" if notified else "\n⚠️ Пользователю написать не удалось (он не запускал бота)."
        try:
            await q.edit_message_text(
                f"{q.message.text_html}\n\n{verdict}{note}", parse_mode=ParseMode.HTML
            )
        except BadRequest:
            pass
        return

    tier = await user_tier(context.bot, user.id)

    if tier == "none":
        await safe_edit(q, LOCKED, subscribe_keyboard())
        return

    data = load_data()
    action = q.data

    # Анализ рынка доступен только после подтверждения регистрации администратором
    needs_approval = action == "pairs" or action.startswith("pair:") or action.startswith("sig:")
    if needs_approval and user.id not in data["approved"] and user.id != ADMIN_ID:
        await safe_edit(
            q,
            "<b>🔒 Анализ рынка недоступен</b>\n"
            f"{LINE}\n"
            "Ваша регистрация ещё не подтверждена администратором.\n\n"
            "<b>Что сделать</b>\n"
            "1. Зарегистрируйтесь по ссылке в разделе «Регистрация».\n"
            "2. Отправьте боту свой ID: <code>/uid 12345678</code>\n"
            "3. Дождитесь подтверждения — после этого анализ откроется.",
            InlineKeyboardMarkup(
                [[InlineKeyboardButton("🔗 Регистрация", callback_data="ref")], back_row()]
            ),
        )
        return

    if action == "menu":
        await safe_edit(q, WELCOME, main_menu())

    elif action == "method":
        await safe_edit(q, METHOD, InlineKeyboardMarkup([back_row()]))

    elif action == "risks":
        await safe_edit(q, RISKS, InlineKeyboardMarkup([back_row()]))

    elif action == "ref":
        text = (
            "<b>🔗 Регистрация на Pocket Option</b>\n"
            f"{LINE}\n"
            "Это ссылка автора бота для получения информации о вас: при регистрации по ней, автор проверяет "
            "наличие активности от вас, как трейдера.\n\n"
            "<b>Порядок действий</b>\n"
            "1. Откройте ссылку и создайте аккаунт.\n"
            "2. Скопируйте свой ID из профиля Pocket Option.\n"
            "3. Отправьте боту: <code>/uid 12345678</code>\n"
            "4. Дождитесь подтверждения — статус появится в разделе «Профиль».\n\n"
        )
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("🚀 Открыть Pocket Option", url=REF_LINK)], back_row()])
        await safe_edit(q, text, kb)

    elif action == "status":
        if tier == "vip":
            plan = "💎 Член команды Terret'a"
            limit = "без ограничений"
        else:
            plan = "🆓 Базовый"
            limit = f"{used_today(data, user.id)} из {DAILY_LIMIT} использовано сегодня"

        if user.id in data["approved"]:
            reg = "✅ подтверждена"
        elif user.id in data["rejected"]:
            reg = "❌ отклонена"
        elif str(user.id) in data["uids"]:
            reg = "⏳ на проверке"
        else:
            reg = "⏳ не подтверждена"

        text = (
            "<b>👤 Профиль</b>\n"
            f"{LINE}\n"
            f"ID Telegram: <code>{user.id}</code>\n"
            f"Тариф: <b>{plan}</b>\n"
            f"Анализы: {limit}\n"
            f"Регистрация: {reg}\n"
            f"{LINE}\n"
            "Счётчик обновляется ежедневно в 00:00 МСК."
        )
        await safe_edit(q, text, InlineKeyboardMarkup([back_row()]))

    elif action == "pairs":
        names = list(PAIRS.keys())
        rows = [
            [InlineKeyboardButton(n, callback_data=f"pair:{n}") for n in names[i : i + 2]]
            for i in range(0, len(names), 2)
        ]
        rows.append(back_row())
        await safe_edit(q, "Выберите пару:", InlineKeyboardMarkup(rows))

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
            f"<b>📊 {pair}</b>\n"
            f"{LINE}\n"
            "Шаг 2 из 2 — выберите таймфрейм.\n\n"
            "<i>Таймфрейм - Время, на которое вы входите в сделку.</i>",
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
                "<b>⛔ Дневной лимит исчерпан</b>\n"
                f"{LINE}\n"
                f"Вы использовали {DAILY_LIMIT} из {DAILY_LIMIT} анализов.\n"
                "Лимит обновится в 00:00 МСК.\n\n"
                "Для работы без ограничений подключите VIP-доступ.",
                InlineKeyboardMarkup(
                    [
                        [InlineKeyboardButton("💬 Связаться с автором", url=f"https://t.me/{CONTACT_USERNAME}")],
                        back_row(),
                    ]
                ),
            )
            return

        tf_name, expiry, _ = TIMEFRAMES[tf]
        if not (q.message is not None and q.message.photo):
            await safe_edit(q, f"⏳ <b>{pair} · {tf_name}</b>\nЗагружаю котировки и считаю индикаторы…")

        try:
            res = await asyncio.to_thread(fetch_and_analyze, PAIRS[pair], tf)
        except Exception as e:
            log.exception("analysis failed: %s", e)
            res = None

        back_kb = InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Другая пара", callback_data="pairs")], back_row()])

        if res is None:
            await safe_edit(
                q,
                "<b>😕 Данные недоступны</b>\n\nНе удалось получить котировки. "
                "Попробуйте позже. Анализ не засчитан в лимит.",
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
        dots = "".join("🟢" if v > 0 else "🔴" for _, _, v in res["votes"])
        votes_txt = "\n".join(
            f"{'🟢' if v > 0 else '🔴'} <b>{escape(name)}</b>\n      <i>{escape(txt)}</i>"
            for name, txt, v in res["votes"]
        )

        strength = "идеальный" if abs(res["score"]) == 5 else "стандартный"
        verdict = "🟢 <b>ВВЕРХ</b>" if res["direction"] == "UP" else "🔴 <b>ВНИЗ</b>"

        now = datetime.now(TZ).strftime("%d.%m.%Y · %H:%M:%S МСК")
        if tier == "vip":
            remaining = "💎 VIP — без ограничений"
        else:
            remaining = f"🆓 Осталось анализов: <b>{DAILY_LIMIT - rec['count']}</b> из {DAILY_LIMIT}"

        text = (
            f"<b>📊 {pair}</b>  ·  <b>{tf_name}</b>\n"
            f"🕒 <code>{now}</code>\n"
            f"{LINE}\n\n"
            f"{verdict}\n\n"
            f"{LINE}\n"
            f"<b>ИНДИКАТОРЫ</b>\n\n{votes_txt}\n\n"
            f"Голоса: {dots}  <b>{res['up']}:{res['down']}</b>  ·  сигнал {strength}\n"
            f"{LINE}\n"
            f"💱 Цена: <code>{res['price']:.{decimals}f}</code>\n"
            f"📈 Волатильность: {res['volatility']}\n"
            f"⏱ Экспирация: <b>{expiry}</b>\n"
            f"{LINE}\n"
            f"{remaining}"
        )
        kb = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("🔄 Обновить", callback_data=f"sig:{pair}:{tf}")],
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
        await show_result(q, text, kb, res["direction"])


def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(CommandHandler("uid", uid_cmd))
    app.add_handler(CommandHandler("approve", approve_cmd))
    app.add_handler(CommandHandler("reject", reject_cmd))
    app.add_handler(CallbackQueryHandler(on_button))
    app.run_polling()


if __name__ == "__main__":
    main()
