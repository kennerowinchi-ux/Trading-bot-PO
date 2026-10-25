import asyncio
import json
import logging
import os
import time
from datetime import datetime
from html import escape
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("bot")

# ======================= НАСТРОЙКИ =======================

BOT_TOKEN = os.environ["BOT_TOKEN"]
FREE_CHANNEL_ID = int(os.environ["FREE_CHANNEL_ID"])
VIP_CHANNEL_ID = int(os.environ["VIP_CHANNEL_ID"])

FREE_CHANNEL_URL = os.getenv(
    "FREE_CHANNEL_URL",
    "https://t.me/+F_OldsGjeK5lMjBi",
)
VIP_CHANNEL_URL = os.getenv(
    "VIP_CHANNEL_URL",
    "https://t.me/+itYuwMLRnBcyNDEy",
)

REF_LINK = os.environ["REF_LINK"]
ADMIN_ID = int(os.getenv("ADMIN_ID") or 0)
CONTACT_USERNAME = os.getenv("CONTACT_USERNAME", "Terret911")

DAILY_LIMIT = 10
TZ = ZoneInfo("Europe/Moscow")
DATA_FILE = "data.json"
CACHE_TTL = 45

with open("pairs.json", encoding="utf-8") as file:
    PAIRS = json.load(file)

# Код -> (название, ориентир по экспирации, максимальный возраст свечи в минутах).
# Порог используется для статуса котировки, но не блокирует анализ.
TIMEFRAMES = {
    "1m": ("M1", "1–3 мин", 10),
    "5m": ("M5", "5–10 мин", 25),
    "15m": ("M15", "15–30 мин", 50),
}

LINE = "━━━━━━━━━━━━━━━━━━"


# ======================= ХРАНИЛИЩЕ =======================

def load_data() -> dict:
    try:
        with open(DATA_FILE, encoding="utf-8") as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    data.setdefault("usage", {})
    data.setdefault("uids", {})
    data.setdefault("approved", [])
    return data


def save_data(data: dict) -> None:
    with open(DATA_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False)


def today() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%d")


def used_today(data: dict, user_id: int) -> int:
    record = data["usage"].get(str(user_id))
    if record and record.get("date") == today():
        return int(record.get("count", 0))
    return 0


# ======================= ПОДПИСКА =======================

async def is_member(bot, chat_id: int, user_id: int) -> bool:
    try:
        member = await bot.get_chat_member(chat_id, user_id)
        return member.status in ("member", "administrator", "creator") or (
            member.status == "restricted"
            and getattr(member, "is_member", False)
        )
    except Exception as error:
        log.warning("get_chat_member failed for %s: %s", chat_id, error)
        return False


async def user_tier(bot, user_id: int) -> str:
    """Возвращает 'vip', 'free' или 'none'."""
    if await is_member(bot, VIP_CHANNEL_ID, user_id):
        return "vip"
    if await is_member(bot, FREE_CHANNEL_ID, user_id):
        return "free"
    return "none"


# ======================= АНАЛИЗ =======================

def analyze_df(df: pd.DataFrame) -> dict:
    """Рассчитывает пять индикаторов и их голоса: вверх, вниз или нейтрально."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    macd = (
        close.ewm(span=12, adjust=False).mean()
        - close.ewm(span=26, adjust=False).mean()
    )
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - macd_signal

    middle = close.rolling(20).mean()
    std = close.rolling(20).std()
    upper = middle + 2 * std
    lower = middle - 2 * std

    lowest_low = low.rolling(14).min()
    highest_high = high.rolling(14).max()
    stoch_range = (highest_high - lowest_low).replace(0, np.nan)
    stochastic = (
        100 * (close - lowest_low) / stoch_range
    ).rolling(3).mean()

    previous_close = close.shift()
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.ewm(alpha=1 / 14, adjust=False).mean()
    atr_average = atr.rolling(50).mean().iloc[-1]

    if pd.notna(atr_average) and atr_average != 0:
        atr_ratio = float(atr.iloc[-1] / atr_average)
    else:
        atr_ratio = 1.0

    votes = []

    e9 = float(ema9.iloc[-1])
    e21 = float(ema21.iloc[-1])
    e50 = float(ema50.iloc[-1])

    if e9 > e21 > e50:
        votes.append(("Тренд · EMA 9/21/50", "восходящая структура", 1))
    elif e9 < e21 < e50:
        votes.append(("Тренд · EMA 9/21/50", "нисходящая структура", -1))
    else:
        votes.append(("Тренд · EMA 9/21/50", "смешанная структура", 0))

    rsi_value = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else 50.0

    if rsi_value < 35:
        votes.append(
            ("Импульс · RSI(14)", f"{rsi_value:.1f} — зона перепроданности", 1)
        )
    elif rsi_value > 65:
        votes.append(
            ("Импульс · RSI(14)", f"{rsi_value:.1f} — зона перекупленности", -1)
        )
    else:
        votes.append(
            ("Импульс · RSI(14)", f"{rsi_value:.1f} — нейтральная зона", 0)
        )

    hist_now = (
        float(histogram.iloc[-1])
        if pd.notna(histogram.iloc[-1])
        else 0.0
    )
    hist_previous = (
        float(histogram.iloc[-2])
        if pd.notna(histogram.iloc[-2])
        else 0.0
    )

    if hist_now > 0 and hist_now > hist_previous:
        votes.append(
            ("Моментум · MACD", "гистограмма выше нуля и растёт", 1)
        )
    elif hist_now < 0 and hist_now < hist_previous:
        votes.append(
            ("Моментум · MACD", "гистограмма ниже нуля и снижается", -1)
        )
    else:
        votes.append(
            ("Моментум · MACD", "нет подтверждённого направления", 0)
        )

    price = float(close.iloc[-1])
    lower_value = float(lower.iloc[-1]) if pd.notna(lower.iloc[-1]) else price
    upper_value = float(upper.iloc[-1]) if pd.notna(upper.iloc[-1]) else price

    if price < lower_value:
        votes.append(
            ("Волатильность · Bollinger", "цена ниже нижней границы", 1)
        )
    elif price > upper_value:
        votes.append(
            ("Волатильность · Bollinger", "цена выше верхней границы", -1)
        )
    else:
        votes.append(
            ("Волатильность · Bollinger", "цена внутри канала", 0)
        )

    stoch_value = (
        float(stochastic.iloc[-1])
        if pd.notna(stochastic.iloc[-1])
        else 50.0
    )

    if stoch_value < 20:
        votes.append(
            ("Осциллятор · Stochastic", f"{stoch_value:.1f} — перепроданность", 1)
        )
    elif stoch_value > 80:
        votes.append(
            ("Осциллятор · Stochastic", f"{stoch_value:.1f} — перекупленность", -1)
        )
    else:
        votes.append(
            ("Осциллятор · Stochastic", f"{stoch_value:.1f} — нейтральная зона", 0)
        )

    score = sum(vote for _, _, vote in votes)

    if score >= 2:
        direction = "UP"
    elif score <= -2:
        direction = "DOWN"
    else:
        direction = "NEUTRAL"

    if atr_ratio < 0.8:
        volatility = "низкая"
    elif atr_ratio > 1.3:
        volatility = "повышенная"
    else:
        volatility = "нормальная"

    return {
        "price": price,
        "votes": votes,
        "score": score,
        "direction": direction,
        "up": sum(1 for _, _, vote in votes if vote > 0),
        "down": sum(1 for _, _, vote in votes if vote < 0),
        "flat": sum(1 for _, _, vote in votes if vote == 0),
        "volatility": volatility,
    }


_cache: dict = {}


def fetch_and_analyze(ticker: str, interval: str) -> dict | None:
    cache_key = (ticker, interval)
    cached = _cache.get(cache_key)

    if cached and time.time() - cached[0] < CACHE_TTL:
        return cached[1]

    df = yf.download(
        ticker,
        period="5d",
        interval=interval,
        progress=False,
        auto_adjust=True,
    )

    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required_columns = {"Close", "High", "Low"}
    if not required_columns.issubset(df.columns):
        return None

    df = df.dropna(subset=["Close", "High", "Low"])
    if len(df) < 60:
        return None

    last_timestamp = df.index[-1]
    if last_timestamp.tzinfo is None:
        last_timestamp = last_timestamp.tz_localize("UTC")
    else:
        last_timestamp = last_timestamp.tz_convert("UTC")

    age_min = (
        pd.Timestamp.now(tz="UTC") - last_timestamp
    ).total_seconds() / 60

    result = analyze_df(df)
    result["age_min"] = max(0.0, age_min)
    result["stale"] = age_min > TIMEFRAMES[interval][2]
    result["last_candle"] = last_timestamp

    _cache[cache_key] = (time.time(), result)
    return result


# ======================= ТЕКСТЫ =======================

WELCOME = (
    "<b>📈 PO SIGNALS</b>\n"
    "<i>Технический терминал для валютных пар</i>\n"
    f"{LINE}\n"
    "Анализ рыночных котировок по пяти индикаторам "
    "и трём таймфреймам.\n\n"
    "<b>Возможности</b>\n"
    "🔎 Разбор выбранной валютной пары\n"
    "📊 EMA · RSI · MACD · Bollinger · Stochastic\n"
    "⏱ Таймфреймы M1 · M5 · M15\n"
    "🧭 Баланс направлений и состояние рынка\n\n"
    "<b>Доступ</b>\n"
    "▫️ Базовый — 10 анализов в сутки\n"
    "▫️ VIP — без дневного лимита\n"
    f"{LINE}\n"
    "Выберите действие:"
)

METHOD = (
    "<b>🧠 МЕТОДИКА АНАЛИЗА</b>\n"
    f"{LINE}\n"
    "<b>01 · Рыночные данные</b>\n"
    "Котировки загружаются из Yahoo Finance. В выходные и праздничные дни "
    "некоторые инструменты могут не обновляться. В этом случае анализ "
    "строится по последним доступным свечам, а их возраст отображается "
    "в результате. Котировки могут отличаться от Pocket Option.\n\n"
    "<b>02 · Пять индикаторов</b>\n"
    "📈 EMA 9/21/50 — структура тренда\n"
    "⚡ RSI(14) — импульс и зоны экстремума\n"
    "〽️ MACD (12/26/9) — динамика момента\n"
    "🎚 Bollinger Bands (20, 2σ) — положение в канале\n"
    "🔄 Stochastic (14, 3) — положение цены в диапазоне\n\n"
    "<b>03 · Баланс голосов</b>\n"
    "Каждый индикатор даёт голос вверх, вниз или нейтрально. Бот суммирует "
    "голоса и отображает распределение. Это техническая оценка данных, "
    "а не рассчитанная статистическая проходимость.\n\n"
    "<b>04 · Итоговое направление</b>\n"
    "При балансе от +2 отображается направление вверх, при балансе от −2 — "
    "вниз. При меньшем перевесе результат отмечается как нейтральный.\n\n"
    "<b>05 · Работа с Pocket Option</b>\n"
    "Выберите пару и таймфрейм, затем сопоставьте результат с графиком "
    "платформы. Экспирация указана как ориентир. Перед использованием "
    "результата проверьте время последней свечи и цену.\n"
    f"{LINE}"
)

LOCKED = (
    "<b>🔒 Доступ к терминалу</b>\n"
    f"{LINE}\n"
    "Для использования бота требуется подписка на канал.\n\n"
    "1️⃣ Подпишитесь на один из каналов.\n"
    "2️⃣ Нажмите «Проверить подписку».\n\n"
    "▫️ Основной канал — 10 анализов в сутки\n"
    "▫️ VIP-канал — без дневного лимита"
)


# ======================= КЛАВИАТУРЫ =======================

def main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("📊 Новый анализ", callback_data="pairs")],
            [
                InlineKeyboardButton("🧠 Методика", callback_data="method"),
                InlineKeyboardButton("👤 Профиль", callback_data="status"),
            ],
            [InlineKeyboardButton("🔗 Подключить доступ", callback_data="ref")],
            [
                InlineKeyboardButton(
                    "💬 Поддержка",
                    url=f"https://t.me/{CONTACT_USERNAME}",
                )
            ],
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


async def safe_edit(
    query,
    text: str,
    keyboard: InlineKeyboardMarkup | None = None,
) -> None:
    try:
        await query.edit_message_text(
            text,
            parse_mode=ParseMode.HTML,
            reply_markup=keyboard,
            disable_web_page_preview=True,
        )
    except BadRequest as error:
        if "not modified" not in str(error).lower():
            raise


# ======================= КОМАНДЫ =======================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tier = await user_tier(context.bot, update.effective_user.id)

    if tier == "none":
        await update.message.reply_text(
            LOCKED,
            parse_mode=ParseMode.HTML,
            reply_markup=subscribe_keyboard(),
        )
        return

    await update.message.reply_text(
        WELCOME,
        parse_mode=ParseMode.HTML,
        reply_markup=main_menu(),
    )


async def uid_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "<b>Подключение доступа</b>\n\n"
            "Отправьте команду:\n"
            "<code>/uid 12345678</code>\n\n"
            "Вместо числа укажите ID из профиля Pocket Option.",
            parse_mode=ParseMode.HTML,
        )
        return

    user = update.effective_user
    pocket_id = context.args[0][:32]

    data = load_data()
    data["uids"][str(user.id)] = pocket_id
    save_data(data)

    await update.message.reply_text(
        "✅ <b>ID получен.</b>\n"
        "Заявка отправлена на проверку.",
        parse_mode=ParseMode.HTML,
    )

    if ADMIN_ID:
        try:
            username = escape(user.username or str(user.id))
            safe_pocket_id = escape(pocket_id)

            await context.bot.send_message(
                ADMIN_ID,
                "🆕 <b>Заявка на подтверждение</b>\n"
                f"Пользователь: @{username} (<code>{user.id}</code>)\n"
                f"ID Pocket Option: <code>{safe_pocket_id}</code>\n\n"
                f"Подтвердить: <code>/approve {user.id}</code>",
                parse_mode=ParseMode.HTML,
            )
        except Exception as error:
            log.warning("Не удалось отправить заявку админу: %s", error)


async def approve_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID or not context.args:
        return

    try:
        user_id = int(context.args[0])
    except ValueError:
        return

    data = load_data()

    if user_id not in data["approved"]:
        data["approved"].append(user_id)
        save_data(data)

    await update.message.reply_text(f"✅ Пользователь {user_id} подтверждён.")

    try:
        await context.bot.send_message(
            user_id,
            "✅ <b>Регистрация подтверждена.</b>",
            parse_mode=ParseMode.HTML,
        )
    except Exception as error:
        log.warning("Не удалось уведомить пользователя: %s", error)


# ======================= КНОПКИ =======================

async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    user = query.from_user
    tier = await user_tier(context.bot, user.id)

    if tier == "none":
        await safe_edit(query, LOCKED, subscribe_keyboard())
        return

    data = load_data()
    action = query.data

    if action == "menu":
        await safe_edit(query, WELCOME, main_menu())
        return

    if action == "method":
        await safe_edit(
            query,
            METHOD,
            InlineKeyboardMarkup([back_row()]),
        )
        return

    if action == "ref":
        text = (
            "<b>🔗 ПОДКЛЮЧЕНИЕ ДОСТУПА</b>\n"
            f"{LINE}\n"
            "Регистрация по ссылке — условие предоставления бесплатного "
            "доступа к терминалу.\n\n"
            "<b>Порядок подключения</b>\n"
            "1️⃣ Откройте ссылку и зарегистрируйтесь.\n"
            "2️⃣ Скопируйте ID из профиля Pocket Option.\n"
            "3️⃣ Отправьте боту: <code>/uid 12345678</code>\n"
            "4️⃣ Дождитесь подтверждения.\n"
            f"{LINE}"
        )

        keyboard = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("🚀 Открыть Pocket Option", url=REF_LINK)],
                back_row(),
            ]
        )
        await safe_edit(query, text, keyboard)
        return

    if action == "status":
        if tier == "vip":
            plan = "💎 VIP"
            limit = "без дневного лимита"
        else:
            plan = "🆓 Базовый"
            limit = (
                f"{used_today(data, user.id)} из {DAILY_LIMIT} "
                "анализов использовано сегодня"
            )

        registration = (
            "✅ подтверждена"
            if user.id in data["approved"]
            else "⏳ ожидает подтверждения"
        )

        text = (
            "<b>👤 ПРОФИЛЬ</b>\n"
            f"{LINE}\n"
            f"Telegram ID: <code>{user.id}</code>\n"
            f"Тариф: <b>{plan}</b>\n"
            f"Анализы: {limit}\n"
            f"Регистрация: {registration}\n"
            f"{LINE}\n"
            "Лимит обновляется ежедневно в 00:00 МСК."
        )

        keyboard_rows = [back_row()]
        if tier != "vip":
            keyboard_rows.insert(
                0,
                [InlineKeyboardButton("💎 Перейти на VIP", url=VIP_CHANNEL_URL)],
            )

        await safe_edit(query, text, InlineKeyboardMarkup(keyboard_rows))
        return

    if action == "pairs":
        names = list(PAIRS.keys())
        rows = [
            [
                InlineKeyboardButton(name, callback_data=f"pair:{name}")
                for name in names[index : index + 2]
            ]
            for index in range(0, len(names), 2)
        ]
        rows.append(back_row())

        await safe_edit(
            query,
            "<b>📊 НОВЫЙ АНАЛИЗ</b>\n"
            f"{LINE}\n"
            "Шаг 1 из 2 — выберите валютную пару:",
            InlineKeyboardMarkup(rows),
        )
        return

    if action.startswith("pair:"):
        pair = action[5:]
        if pair not in PAIRS:
            return

        buttons = [
            InlineKeyboardButton(
                f"{name} · экспирация {expiry}",
                callback_data=f"sig:{pair}:{timeframe}",
            )
            for timeframe, (name, expiry, _) in TIMEFRAMES.items()
        ]

        keyboard = [[button] for button in buttons]
        keyboard.append(
            [InlineKeyboardButton("⬅️ К выбору пары", callback_data="pairs")]
        )

        await safe_edit(
            query,
            f"<b>📊 {escape(pair)}</b>\n"
            f"{LINE}\n"
            "Шаг 2 из 2 — выберите таймфрейм.\n\n"
            "⏱ На малых таймфреймах больше краткосрочных колебаний.",
            InlineKeyboardMarkup(keyboard),
        )
        return

    if not action.startswith("sig:"):
        return

    try:
        _, pair, timeframe = action.split(":")
    except ValueError:
        return

    if pair not in PAIRS or timeframe not in TIMEFRAMES:
        return

    if tier == "free" and used_today(data, user.id) >= DAILY_LIMIT:
        keyboard = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("💎 VIP-канал", url=VIP_CHANNEL_URL)],
                [
                    InlineKeyboardButton(
                        "💬 Поддержка",
                        url=f"https://t.me/{CONTACT_USERNAME}",
                    )
                ],
                back_row(),
            ]
        )
        await safe_edit(
            query,
            "<b>⛔ ДНЕВНОЙ ЛИМИТ ИСЧЕРПАН</b>\n"
            f"{LINE}\n"
            f"Использовано {DAILY_LIMIT} из {DAILY_LIMIT} анализов.\n"
            "Лимит обновится в 00:00 МСК.",
            keyboard,
        )
        return

    timeframe_name, expiry, _ = TIMEFRAMES[timeframe]

    await safe_edit(
        query,
        f"⏳ <b>{escape(pair)} · {timeframe_name}</b>\n"
        "Загружаю котировки и рассчитываю индикаторы…",
    )

    try:
        result = await asyncio.to_thread(
            fetch_and_analyze,
            PAIRS[pair],
            timeframe,
        )
    except Exception as error:
        log.exception("Analysis failed: %s", error)
        result = None

    back_keyboard = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🔄 Другая пара", callback_data="pairs")],
            back_row(),
        ]
    )

    if result is None:
        await safe_edit(
            query,
            "<b>😕 КОТИРОВКИ НЕДОСТУПНЫ</b>\n\n"
            "Не удалось получить достаточное количество данных. "
            "Попробуйте позже. Анализ не засчитан в лимит.",
            back_keyboard,
        )
        return

    record = data["usage"].get(str(user.id))
    if not record or record.get("date") != today():
        record = {"date": today(), "count": 0}

    record["count"] += 1
    data["usage"][str(user.id)] = record
    save_data(data)

    decimals = 3 if "JPY" in pair else 5
    marks = {1: "▲", -1: "▼", 0: "●"}

    votes_text = "\n".join(
        f"{marks[vote]} <b>{escape(name)}</b>\n    {escape(description)}"
        for name, description, vote in result["votes"]
    )

    if result["direction"] == "UP":
        strength = {2: "умеренный", 3: "выраженный"}.get(
            abs(result["score"]),
            "сильный",
        )
        heading = f"🟢 <b>ВВЕРХ (CALL)</b> · {strength} перевес"
    elif result["direction"] == "DOWN":
        strength = {2: "умеренный", 3: "выраженный"}.get(
            abs(result["score"]),
            "сильный",
        )
        heading = f"🔴 <b>ВНИЗ (PUT)</b> · {strength} перевес"
    else:
        heading = (
            "⚪ <b>НЕЙТРАЛЬНОЕ НАПРАВЛЕНИЕ</b>\n"
            "Голоса индикаторов не сформировали достаточный перевес."
        )

    if result["stale"]:
        quote_status = (
            f"🕰 Последняя свеча: {result['age_min'] / 60:.1f} ч назад "
            "(используются последние доступные данные)"
        )
    else:
        quote_status = f"🟢 Данные обновлялись около {result['age_min']:.0f} мин назад"

    now = datetime.now(TZ).strftime("%d.%m.%Y %H:%M МСК")

    if tier == "vip":
        remaining = "💎 VIP — без дневного лимита"
    else:
        remaining = (
            f"🆓 Осталось анализов сегодня: "
            f"{DAILY_LIMIT - record['count']} из {DAILY_LIMIT}"
        )

    total_votes = len(result["votes"])
    up_share = result["up"] / total_votes * 100 if total_votes else 0
    down_share = result["down"] / total_votes * 100 if total_votes else 0

    text = (
        f"<b>📊 {escape(pair)} · {timeframe_name}</b>\n"
        f"<code>{now}</code>\n"
        f"{LINE}\n"
        f"<b>АНАЛИТИЧЕСКИЙ РЕЗУЛЬТАТ</b>\n"
        f"{heading}\n\n"
        f"Баланс голосов: <b>{result['score']:+d}</b>\n"
        f"▲ Вверх: {result['up']} · "
        f"▼ Вниз: {result['down']} · "
        f"● Нейтрально: {result['flat']}\n"
        f"Доли от пяти индикаторов: ▲ {up_share:.0f}% · ▼ {down_share:.0f}%\n"
        f"{LINE}\n"
        f"<b>ИНДИКАТОРЫ</b>\n{votes_text}\n"
        f"{LINE}\n"
        f"<b>СОСТОЯНИЕ РЫНКА</b>\n"
        f"Цена: <code>{result['price']:.{decimals}f}</code>\n"
        f"Волатильность ATR: {result['volatility']}\n"
        f"Ориентир по экспирации: {expiry}\n"
        f"{quote_status}\n"
        f"{LINE}\n"
        f"{remaining}"
    )

    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "🔄 Обновить анализ",
                    callback_data=f"sig:{pair}:{timeframe}",
                )
            ],
            [
                InlineKeyboardButton(
                    "🕒 Другой таймфрейм",
                    callback_data=f"pair:{pair}",
                ),
                InlineKeyboardButton(
                    "💱 Другая пара",
                    callback_data="pairs",
                ),
            ],
            [
                InlineKeyboardButton("🧠 Методика", callback_data="method"),
                InlineKeyboardButton("⬅️ Меню", callback_data="menu"),
            ],
        ]
    )

    await safe_edit(query, text, keyboard)


# ======================= ЗАПУСК =======================

def main():
    application = Application.builder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler(["start", "help"], start))
    application.add_handler(CommandHandler("uid", uid_cmd))
    application.add_handler(CommandHandler("approve", approve_cmd))
    application.add_handler(CallbackQueryHandler(on_button))

    application.run_polling()


if __name__ == "__main__":
    main()
