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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bot")

# ======================= НАСТРОЙКИ =======================
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
CACHE_TTL = 45  # секунд

with open("pairs.json", encoding="utf-8") as f:
    PAIRS = json.load(f)  # Например: {"EUR/USD": "EURUSD=X"}

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
        member = await bot.get_chat_member(chat_id, user_id)
        return member.status in ("member", "administrator", "creator") or (
            member.status == "restricted" and getattr(member, "is_member", False)
        )
    except Exception as e:
        log.warning("get_chat_member failed for %s: %s", chat_id, e)
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
    """Рассчитывает пять индикаторов; каждый даёт голос вверх, вниз или нейтрально."""
    close, high, low = df["Close"], df["High"], df["Low"]

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()

    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    upper, lower = mid + 2 * std, mid - 2 * std

    ll, hh = low.rolling(14).min(), high.rolling(14).max()
    stoch = (100 * (close - ll) / (hh - ll).replace(0, np.nan)).rolling(3).mean()

    prev_close = close.shift()
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    atr_average = atr.rolling(50).mean().iloc[-1]
    atr_ratio = float(atr.iloc[-1] / atr_average) if pd.notna(atr_average) and atr_average != 0 else 1.0

    votes = []

    e9, e21, e50 = float(ema9.iloc[-1]), float(ema21.iloc[-1]), float(ema50.iloc[-1])
    if e9 > e21 > e50:
        votes.append(("Тренд · EMA 9/21/50", "восходящая структура", 1))
    elif e9 < e21 < e50:
        votes.append(("Тренд · EMA 9/21/50", "нисходящая структура", -1))
    else:
        votes.append(("Тренд · EMA 9/21/50", "смешанная структура", 0))

    r = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else 50.0
    if r < 35:
        votes.append(("Импульс · RSI(14)", f"{r:.1f} — зона перепроданности", 1))
    elif r > 65:
        votes.append(("Импульс · RSI(14)", f"{r:.1f} — зона перекупленности", -1))
    else:
        votes.append(("Импульс · RSI(14)", f"{r:.1f} — нейтральная зона", 0))

    h_now = float(hist.iloc[-1]) if pd.notna(hist.iloc[-1]) else 0.0
    h_prev = float(hist.iloc[-2]) if pd.notna(hist.iloc[-2]) else 0.0
    if h_now > 0 and h_now > h_prev:
        votes.append(("Моментум · MACD", "гистограмма > 0 и растёт", 1))
    elif h_now < 0 and h_now < h_prev:
        votes.append(("Моментум · MACD", "гистограмма < 0 и падает", -1))
    else:
        votes.append(("Моментум · MACD", "нет подтверждённого направления", 0))

    price = float(close.iloc[-1])
    lower_value = float(lower.iloc[-1]) if pd.notna(lower.iloc[-1]) else price
    upper_value = float(upper.iloc[-1]) if pd.notna(upper.iloc[-1]) else price
    if price < lower_value:
        votes.append(("Волатильность · Bollinger", "цена ниже нижней границы", 1))
    elif price > upper_value:
        votes.append(("Волатильность · Bollinger", "цена выше верхней границы", -1))
    else:
        votes.append(("Волатильность · Bollinger", "цена внутри канала", 0))

    s = float(stoch.iloc[-1]) if pd.notna(stoch.iloc[-1]) else 50.0
    if s < 20:
        votes.append(("Осциллятор · Stochastic", f"{s:.1f} — перепроданность", 1))
    elif s > 80:
        votes.append(("Осциллятор · Stochastic", f"{s:.1f} — перекупленность", -1))
    else:
        votes.append(("Осциллятор · Stochastic", f"{s:.1f} — нейтральная зона", 0))

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
    key = (ticker, interval)
    cached = _cache.get(key)
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
    "Технический анализ валютных пар для работы на Pocket Option. "
    "Бот получает рыночные котировки, рассчитывает пять индикаторов "
    "и показывает их направления и общий баланс по выбранному таймфрейму.\n\n"
    "<b>Тарифы доступа</b>\n"
    "▫️ Базовый — <b>10 анализов в сутки</b>\n"
    "▫️ VIP — <b>без ограничений</b>\n"
    f"{LINE}\n"
    "Выберите раздел:"
)

METHOD = (
    "<b>📚 Методика анализа для Pocket Option</b>\n"
    f"{LINE}\n"
    "<b>1. Рыночные данные.</b> Бот загружает котировки валютных пар и формирует "
    "анализ для выбранного таймфрейма M1, M5 или M15. Котировки поступают из "
    "Yahoo Finance; они могут отличаться от котировок Pocket Option. "
    "Сопоставляйте время свечей и цену на платформе перед использованием сигнала.\n\n"
    "<b>2. Индикаторы.</b> Рассчитываются пять технических показателей:\n"
    "▫️ EMA 9/21/50 — структура тренда\n"
    "▫️ RSI(14) — зоны перекупленности и перепроданности\n"
    "▫️ MACD (12/26/9) — направление и изменение моментума\n"
    "▫️ Bollinger Bands (20, 2σ) — положение цены относительно канала\n"
    "▫️ Stochastic (14, 3) — положение цены в диапазоне\n\n"
    "<b>3. Как читать соотношение голосов.</b> Каждый индикатор формирует один голос: "
    "вверх, вниз или нейтрально. Бот показывает распределение этих голосов — "
    "например, 2 из 5 за рост и 2 из 5 за снижение, а оставшийся голос нейтральный. "
    "Это соотношение мнений индикаторов, а не процент вероятности и не заданное "
    "заранее распределение сигналов.\n\n"
    "<b>4. Итоговое направление.</b> Баланс рассчитывается как число голосов вверх "
    "минус число голосов вниз. При балансе от +2 бот показывает направление вверх, "
    "при балансе от −2 — вниз. При меньшем перевесе указывается, что индикаторы "
    "не сформировали единого направления.\n\n"
    "<b>5. Применение на Pocket Option.</b> Выберите пару и таймфрейм, затем "
    "сравните результат с текущим графиком платформы. Экспирация в сообщении "
    "служит ориентиром, соответствующим выбранному таймфрейму.\n"
    f"{LINE}"
)


# ======================= КЛАВИАТУРЫ =======================
def main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("📊 Новый анализ", callback_data="pairs")],
            [
                InlineKeyboardButton("📚 Методика", callback_data="method"),
                InlineKeyboardButton("👤 Профиль", callback_data="status"),
            ],
            [InlineKeyboardButton("🔗 Регистрация", callback_data="ref")],
            [InlineKeyboardButton("💬 Поддержка", url=f"https://t.me/{CONTACT_USERNAME}")],
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
            text,
            parse_mode=ParseMode.HTML,
            reply_markup=kb,
            disable_web_page_preview=True,
        )
    except BadRequest as e:
        if "not modified" not in str(e).lower():
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
    """Пользователь отправляет свой ID Pocket Option, администратор подтверждает его вручную."""
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
    save_data(data)

    await update.message.reply_text(
        "✅ <b>ID получен.</b>\n"
        "Заявка отправлена на проверку, статус появится в разделе «Профиль».",
        parse_mode=ParseMode.HTML,
    )

    if ADMIN_ID:
        try:
            await context.bot.send_message(
                ADMIN_ID,
                f"🆕 <b>Заявка на подтверждение</b>\n"
                f"Пользователь: @{escape(user.username or str(user.id))} "
                f"(<code>{user.id}</code>)\n"
                f"ID Pocket Option: <code>{escape(uid)}</code>\n\n"
                f"Подтвердить: <code>/approve {user.id}</code>",
                parse_mode=ParseMode.HTML,
            )
        except Exception as e:
            log.warning("Не удалось отправить заявку админу: %s", e)


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
            "✅ <b>Регистрация подтверждена.</b> Спасибо!",
            parse_mode=ParseMode.HTML,
        )
    except Exception as e:
        log.warning("Не удалось уведомить пользователя: %s", e)


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

    elif action == "method":
        await safe_edit(query, METHOD, InlineKeyboardMarkup([back_row()]))

    elif action == "ref":
        text = (
            "<b>🔗 Регистрация на Pocket Option</b>\n"
            f"{LINE}\n"
            "Это партнёрская ссылка автора бота: при регистрации по ней автор может "
            "получать вознаграждение от брокера. Для пользователя условия не меняются.\n\n"
            "<b>Порядок действий</b>\n"
            "1. Откройте ссылку и создайте аккаунт.\n"
            "2. Скопируйте свой ID из профиля Pocket Option.\n"
            "3. Отправьте боту: <code>/uid 12345678</code>\n"
            "4. Дождитесь подтверждения — статус появится в разделе «Профиль»."
        )
        keyboard = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("🚀 Открыть Pocket Option", url=REF_LINK)],
                back_row(),
            ]
        )
        await safe_edit(query, text, keyboard)

    elif action == "status":
        if tier == "vip":
            plan = "💎 VIP"
            limit = "без ограничений"
        else:
            plan = "🆓 Базовый"
            limit = f"{used_today(data, user.id)} из {DAILY_LIMIT} использовано сегодня"

        registration = "✅ подтверждена" if user.id in data["approved"] else "⏳ не подтверждена"
        text = (
            "<b>👤 Профиль</b>\n"
            f"{LINE}\n"
            f"ID Telegram: <code>{user.id}</code>\n"
            f"Тариф: <b>{plan}</b>\n"
            f"Анализы: {limit}\n"
            f"Регистрация: {registration}\n"
            f"{LINE}\n"
            "Счётчик обновляется ежедневно в 00:00 МСК."
        )

        keyboard_rows = [back_row()]
        if tier != "vip":
            keyboard_rows.insert(
                0,
                [InlineKeyboardButton("💎 Перейти на VIP", url=VIP_CHANNEL_URL)],
            )

        await safe_edit(query, text, InlineKeyboardMarkup(keyboard_rows))

    elif action == "pairs":
        names = list(PAIRS.keys())
        rows = [
            [InlineKeyboardButton(name, callback_data=f"pair:{name}") for name in names[i : i + 2]]
            for i in range(0, len(names), 2)
        ]
        rows.append(back_row())

        await safe_edit(
            query,
            "<b>📊 Новый анализ</b>\n"
            f"{LINE}\n"
            "Шаг 1 из 2 — выберите валютную пару:",
            InlineKeyboardMarkup(rows),
        )

    elif action.startswith("pair:"):
        pair = action[5:]
        if pair not in PAIRS:
            return

        buttons = [
            InlineKeyboardButton(
                f"{name} · экспирация {expiry}",
                callback_data=f"sig:{pair}:{code}",
            )
            for code, (name, expiry, _) in TIMEFRAMES.items()
        ]
        keyboard = [[button] for button in buttons]
        keyboard.append(
            [InlineKeyboardButton("⬅️ К выбору пары", callback_data="pairs")]
        )

        await safe_edit(
            query,
            f"<b>📊 {pair}</b>\n"
            f"{LINE}\n"
            "Шаг 2 из 2 — выберите таймфрейм.\n\n"
            "<i>Чем меньше таймфрейм, тем больше рыночного шума.</i>",
            InlineKeyboardMarkup(keyboard),
        )

    elif action.startswith("sig:"):
        try:
            _, pair, timeframe = action.split(":")
        except ValueError:
            return

        if pair not in PAIRS or timeframe not in TIMEFRAMES:
            return

        if tier == "free" and used_today(data, user.id) >= DAILY_LIMIT:
            await safe_edit(
                query,
                "<b>⛔ Дневной лимит исчерпан</b>\n"
                f"{LINE}\n"
                f"Вы использовали {DAILY_LIMIT} из {DAILY_LIMIT} анализов.\n"
                "Лимит обновится в 00:00 МСК.\n\n"
                "Для работы без ограничений подключите VIP-доступ.",
                InlineKeyboardMarkup(
                    [
                        [InlineKeyboardButton("💎 VIP-канал", url=VIP_CHANNEL_URL)],
                        [
                            InlineKeyboardButton(
                                "💬 Связаться с автором",
                                url=f"https://t.me/{CONTACT_USERNAME}",
                            )
                        ],
                        back_row(),
                    ]
                ),
            )
            return

        timeframe_name, expiry, _ = TIMEFRAMES[timeframe]
        await safe_edit(
            query,
            f"⏳ <b>{pair} · {timeframe_name}</b>\n"
            "Загружаю котировки и рассчитываю индикаторы…",
        )

        try:
            result = await asyncio.to_thread(
                fetch_and_analyze,
                PAIRS[pair],
                timeframe,
            )
        except Exception as e:
            log.exception("analysis failed: %s", e)
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
                "<b>😕 Данные недоступны</b>\n\n"
                "Не удалось получить котировки. Попробуйте позже. "
                "Анализ не засчитан в лимит.",
                back_keyboard,
            )
            return

        if result["stale"]:
            await safe_edit(
                query,
                "<b>🕒 Нет свежих котировок</b>\n"
                f"{LINE}\n"
                f"Последняя доступная котировка по {pair} получена "
                f"{result['age_min'] / 60:.1f} ч назад. "
                "Анализ по устаревшим данным не выполняется.\n\n"
                "Для валютного рынка котировки обычно доступны с понедельника "
                "по пятницу. Анализ не засчитан в лимит.",
                back_keyboard,
            )
            return

        record = data["usage"].get(str(user.id))
        if not record or record["date"] != today():
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
            strength = {2: "слабый", 3: "умеренный"}.get(
                abs(result["score"]), "сильный"
            )
            heading = f"🟢 <b>ВВЕРХ (CALL)</b> · {strength} перевес"
        elif result["direction"] == "DOWN":
            strength = {2: "слабый", 3: "умеренный"}.get(
                abs(result["score"]), "сильный"
            )
            heading = f"🔴 <b>ВНИЗ (PUT)</b> · {strength} перевес"
        else:
            heading = (
                "⚪ <b>ЧЁТКОГО НАПРАВЛЕНИЯ НЕТ</b>\n"
                "Голоса индикаторов распределились без достаточного перевеса."
            )

        now = datetime.now(TZ).strftime("%d.%m.%Y %H:%M МСК")

        if tier == "vip":
            remaining = "💎 VIP — без ограничений"
        else:
            remaining = (
                f"🆓 Осталось анализов сегодня: "
                f"{DAILY_LIMIT - record['count']} из {DAILY_LIMIT}"
            )

        total_votes = len(result["votes"])
        up_share = result["up"] / total_votes * 100 if total_votes else 0
        down_share = result["down"] / total_votes * 100 if total_votes else 0

        text = (
            f"<b>📊 {pair} · {timeframe_name}</b>\n"
            f"<code>{now}</code>\n"
            f"{LINE}\n"
            f"<b>РЕЗУЛЬТАТ</b>\n{heading}\n\n"
            f"Баланс голосов: <b>{result['score']:+d}</b>  "
            f"(▲ {result['up']} · ▼ {result['down']} · ● {result['flat']})\n"
            f"Доля голосов от пяти индикаторов: "
            f"▲ {up_share:.0f}% · ▼ {down_share:.0f}%\n"
            f"{LINE}\n"
            f"<b>ИНДИКАТОРЫ</b>\n{votes_text}\n"
            f"{LINE}\n"
            f"<b>РЫНОК</b>\n"
            f"Цена: <code>{result['price']:.{decimals}f}</code>\n"
            f"Волатильность (ATR): {result['volatility']}\n"
            f"Ориентир по экспирации: {expiry}\n"
            f"{LINE}\n"
            f"{remaining}"
        )

        keyboard = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("🔄 Обновить", callback_data=f"sig:{pair}:{timeframe}")],
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
                    InlineKeyboardButton("📚 Методика", callback_data="method"),
                    InlineKeyboardButton("⬅️ Меню", callback_data="menu"),
                ],
            ]
        )

        await safe_edit(query, text, keyboard)


def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(CommandHandler("uid", uid_cmd))
    app.add_handler(CommandHandler("approve", approve_cmd))
    app.add_handler(CallbackQueryHandler(on_button))
    app.run_polling()


if __name__ == "__main__":
    main()
