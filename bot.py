import json
import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bot")

# ---------- Настройки (все берутся из переменных окружения / секретов GitHub) ----------
BOT_TOKEN = os.environ["BOT_TOKEN"]
FREE_CHANNEL_ID = int(os.environ["FREE_CHANNEL_ID"])  # например -1001234567890
VIP_CHANNEL_ID = int(os.environ["VIP_CHANNEL_ID"])
FREE_CHANNEL_URL = os.getenv("FREE_CHANNEL_URL", "https://t.me/+F_OldsGjeK5lMjBi")
VIP_CHANNEL_URL = os.getenv("VIP_CHANNEL_URL", "https://t.me/+itYuwMLRnBcyNDEy")
REF_LINK = os.environ["REF_LINK"]  # ваша партнёрская ссылка Pocket Option
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))  # ваш числовой Telegram ID
CONTACT_USERNAME = os.getenv("CONTACT_USERNAME", "Terret911")
DAILY_LIMIT = 10
TZ = ZoneInfo("Europe/Moscow")
DATA_FILE = "data.json"

with open("pairs.json", encoding="utf-8") as f:
    PAIRS = json.load(f)  # {"EUR/USD": "EURUSD=X", ...}

DISCLAIMER = (
    "⚠️ <i>Это не финансовая рекомендация. Расчёт — простой технический индикатор "
    "(EMA + RSI) по рыночным котировкам, он не гарантирует результат. "
    "Торговля бинарными опционами связана с высоким риском потери средств.</i>"
)


# ---------- Простое хранилище ----------
def load_data() -> dict:
    try:
        with open(DATA_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"usage": {}, "uids": {}, "approved": []}


def save_data(data: dict) -> None:
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def today() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%d")


# ---------- Проверка подписки ----------
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


def used_today(data: dict, user_id: int) -> int:
    rec = data["usage"].get(str(user_id))
    return rec["count"] if rec and rec["date"] == today() else 0


# ---------- Расчёт сигнала ----------
def compute_signal(ticker: str) -> dict | None:
    df = yf.download(ticker, period="1d", interval="5m", progress=False, auto_adjust=True)
    if df is None or df.empty:
        df = yf.download(ticker, period="5d", interval="5m", progress=False, auto_adjust=True)
    if df is None or len(df) < 30:
        return None
    close = df["Close"].squeeze().dropna()
    ema_fast = close.ewm(span=9, adjust=False).mean()
    ema_slow = close.ewm(span=21, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, pd.NA))
    r = float(rsi.iloc[-1])
    trend_up = float(ema_fast.iloc[-1]) > float(ema_slow.iloc[-1])
    if trend_up and r < 70:
        direction = "UP"
    elif not trend_up and r > 30:
        direction = "DOWN"
    else:
        direction = "NEUTRAL"
    return {"price": float(close.iloc[-1]), "rsi": r, "trend_up": trend_up, "direction": direction}


# ---------- Клавиатуры и тексты ----------
def main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("📊 Получить сигнал", callback_data="pairs")],
            [
                InlineKeyboardButton("🔗 Регистрация", callback_data="ref"),
                InlineKeyboardButton("👤 Мой статус", callback_data="status"),
            ],
            [InlineKeyboardButton("💬 Связаться со мной", url=f"https://t.me/{CONTACT_USERNAME}")],
        ]
    )


def back_row():
    return [InlineKeyboardButton("⬅️ В меню", callback_data="menu")]


def subscribe_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("📢 Основной канал", url=FREE_CHANNEL_URL)],
            [InlineKeyboardButton("💎 VIP канал", url=VIP_CHANNEL_URL)],
            [InlineKeyboardButton("✅ Я подписался", callback_data="menu")],
        ]
    )


WELCOME = (
    "👋 <b>Добро пожаловать!</b>\n\n"
    "📈 Здесь вы можете получать технические сигналы по валютным парам "
    "<b>Pocket Option</b>.\n\n"
    "🆓 Обычный доступ: <b>10 сигналов в день</b>\n"
    "💎 VIP доступ: <b>без ограничений</b>\n\n"
    "Выберите действие 👇"
)


# ---------- Хэндлеры ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    tier = await user_tier(context.bot, user.id)
    if tier == "none":
        await update.message.reply_text(
            "🔒 <b>Доступ закрыт</b>\n\nЧтобы пользоваться ботом, подпишитесь на канал "
            "и нажмите «Я подписался».",
            parse_mode=ParseMode.HTML,
            reply_markup=subscribe_keyboard(),
        )
        return
    await update.message.reply_text(WELCOME, parse_mode=ParseMode.HTML, reply_markup=main_menu())


async def uid_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Пользователь присылает свой ID на Pocket Option, админ подтверждает вручную."""
    if not context.args:
        await update.message.reply_text(
            "Отправьте: <code>/uid 12345678</code> — ваш ID на Pocket Option.",
            parse_mode=ParseMode.HTML,
        )
        return
    user = update.effective_user
    data = load_data()
    data["uids"][str(user.id)] = context.args[0]
    save_data(data)
    await update.message.reply_text("✅ ID получен, ожидайте подтверждения.")
    if ADMIN_ID:
        await context.bot.send_message(
            ADMIN_ID,
            f"🆕 Заявка: @{user.username or user.id} (id {user.id}), PO ID: {context.args[0]}\n"
            f"Подтвердить: /approve {user.id}",
        )


async def approve_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID or not context.args:
        return
    data = load_data()
    uid = int(context.args[0])
    if uid not in data["approved"]:
        data["approved"].append(uid)
        save_data(data)
    await update.message.reply_text(f"✅ {uid} подтверждён")
    await context.bot.send_message(uid, "✅ Регистрация подтверждена. Спасибо!")


async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    user = q.from_user
    tier = await user_tier(context.bot, user.id)

    if tier == "none":
        await q.edit_message_text(
            "🔒 <b>Доступ закрыт</b>\n\nПодпишитесь на канал и нажмите «Я подписался».",
            parse_mode=ParseMode.HTML,
            reply_markup=subscribe_keyboard(),
        )
        return

    data = load_data()
    action = q.data

    if action == "menu":
        await q.edit_message_text(WELCOME, parse_mode=ParseMode.HTML, reply_markup=main_menu())

    elif action == "ref":
        text = (
            "🔗 <b>Регистрация на Pocket Option</b>\n\n"
            f"Это моя партнёрская ссылка: {REF_LINK}\n\n"
            "После регистрации отправьте свой ID командой:\n"
            "<code>/uid 12345678</code>"
        )
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("🚀 Открыть", url=REF_LINK)], back_row()])
        await q.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=kb)

    elif action == "status":
        if tier == "vip":
            lim = "💎 VIP — безлимит"
        else:
            lim = f"🆓 Обычный — использовано {used_today(data, user.id)}/{DAILY_LIMIT} за сегодня"
        reg = "✅ подтверждена" if user.id in data["approved"] else "⏳ не подтверждена"
        await q.edit_message_text(
            f"👤 <b>Ваш статус</b>\n\n{lim}\n📝 Регистрация: {reg}",
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup([back_row()]),
        )

    elif action == "pairs":
        names = list(PAIRS.keys())
        rows = []
        for i in range(0, len(names), 2):
            rows.append(
                [InlineKeyboardButton(f"💱 {n}", callback_data=f"sig:{n}") for n in names[i : i + 2]]
            )
        rows.append(back_row())
        await q.edit_message_text(
            "💱 <b>Выберите валютную пару:</b>",
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup(rows),
        )

    elif action.startswith("sig:"):
        pair = action[4:]
        if pair not in PAIRS:
            return
        if tier == "free" and used_today(data, user.id) >= DAILY_LIMIT:
            await q.edit_message_text(
                "⛔ <b>Лимит на сегодня исчерпан</b> (10/10).\n\n"
                "💎 Оформите VIP для безлимитного доступа или возвращайтесь завтра.",
                parse_mode=ParseMode.HTML,
                reply_markup=InlineKeyboardMarkup(
                    [
                        [InlineKeyboardButton("💎 VIP канал", url=VIP_CHANNEL_URL)],
                        [InlineKeyboardButton("💬 Связаться", url=f"https://t.me/{CONTACT_USERNAME}")],
                        back_row(),
                    ]
                ),
            )
            return

        await q.edit_message_text(f"⏳ Загружаю котировки <b>{pair}</b>...", parse_mode=ParseMode.HTML)
        sig = compute_signal(PAIRS[pair])
        if sig is None:
            await q.edit_message_text(
                "😕 Не удалось получить котировки (рынок закрыт или источник недоступен). "
                "Попробуйте позже.",
                reply_markup=InlineKeyboardMarkup([back_row()]),
            )
            return

        rec = data["usage"].get(str(user.id))
        if not rec or rec["date"] != today():
            rec = {"date": today(), "count": 0}
        rec["count"] += 1
        data["usage"][str(user.id)] = rec
        save_data(data)

        icon = {"UP": "🟢 ВВЕРХ ⬆️", "DOWN": "🔴 ВНИЗ ⬇️", "NEUTRAL": "⚪ НЕЙТРАЛЬНО — лучше пропустить"}[
            sig["direction"]
        ]
        trend = "восходящий 📈" if sig["trend_up"] else "нисходящий 📉"
        text = (
            f"📊 <b>{pair}</b>\n"
            f"━━━━━━━━━━━━━━\n"
            f"💰 Цена: <code>{sig['price']:.5f}</code>\n"
            f"📐 Тренд (EMA 9/21): {trend}\n"
            f"📉 RSI(14): <code>{sig['rsi']:.1f}</code>\n"
            f"━━━━━━━━━━━━━━\n"
            f"🎯 Сигнал: <b>{icon}</b>\n"
            f"⏱ Таймфрейм: 5 мин\n\n{DISCLAIMER}"
        )
        kb = InlineKeyboardMarkup(
            [[InlineKeyboardButton("🔄 Другая пара", callback_data="pairs")], back_row()]
        )
        await q.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=kb)


def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("uid", uid_cmd))
    app.add_handler(CommandHandler("approve", approve_cmd))
    app.add_handler(CallbackQueryHandler(on_button))
    app.run_polling()


if __name__ == "__main__":
    main()
