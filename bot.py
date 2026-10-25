import asyncio
import json
import logging
import os
import random
from datetime import datetime, timedelta
from html import escape
from zoneinfo import ZoneInfo

import yfinance as yf
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

TZ = ZoneInfo("Europe/Moscow")

ASSETS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "CAD=X",
    "USD/CHF": "CHF=X",
    "EUR/JPY": "EURJPY=X",
    "EUR/GBP": "EURGBP=X",
}

AWAITING_UID = 1

BOT_TOKEN = os.getenv("BOT_TOKEN")
REF_LINK = os.getenv("REF_LINK", "https://pocketoption.com")
ADMIN_ID = int(os.getenv("ADMIN_ID") or 0)

# ВНИМАНИЕ: в GitHub Actions файловая система эфемерная — после перезапуска список
# одобренных пропадёт. Для постоянного хранения нужен хостинг с диском или внешняя БД.
APPROVED_USERS_FILE = "approved_users.json"

DISCLAIMER = (
    "<i>⚠️ Упрощённая оценка, реальная вероятность ≈ 50/50. Не инвестиционная "
    "рекомендация, результат не гарантирован. Бинарные опционы — высокий риск потери средств.</i>"
)


def get_approved_users() -> list:
    if os.path.exists(APPROVED_USERS_FILE):
        try:
            with open(APPROVED_USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []


def approve_user_id(user_id: int) -> None:
    users = get_approved_users()
    if user_id not in users:
        users.append(user_id)
        with open(APPROVED_USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f)


def is_allowed(user_id: int) -> bool:
    return user_id == ADMIN_ID or user_id in get_approved_users()


def generate_market_signal(ticker_symbol: str):
    """Направление по RSI(14) на 5-минутках. Если данных нет — случайный выбор.
    Возвращает (направление, rsi или None, источник)."""
    try:
        df = yf.Ticker(ticker_symbol).history(period="1d", interval="5m")
        if df.empty or len(df) < 20:
            raise ValueError("Недостаточно данных")
        delta = df["Close"].diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        rsi = 100 - 100 / (1 + gain / (loss + 1e-10))
        current = float(rsi.iloc[-1])
        direction = "🟢 ВВЕРХ (CALL)" if current > 50 else "🔴 ВНИЗ (PUT)"
        return direction, current, "RSI(14) по рыночным котировкам"
    except Exception as e:
        logger.error("Ошибка yfinance для %s: %s", ticker_symbol, e)
        direction = random.choice(["🟢 ВВЕРХ (CALL)", "🔴 ВНИЗ (PUT)"])
        return direction, None, "случайный выбор (нет рыночных данных)"


# ---------- меню ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    text = (
        f"👋 <b>Привет, {escape(user.first_name or '')}!</b>\n\n"
        "🤖 Бот для <b>Pocket Option</b>: сигналы в режиме «50/50» на основе простого "
        "технического индикатора.\n\n"
        "Чтобы получить доступ, активируйте аккаунт по партнёрской ссылке автора "
        "(он может получать вознаграждение от брокера).\n\n"
        f"{DISCLAIMER}"
    )
    keyboard = [
        [InlineKeyboardButton("🔗 1. Регистрация аккаунта", url=REF_LINK)],
        [InlineKeyboardButton("🆔 2. Подтвердить свой UID", callback_data="verify_uid")],
        [InlineKeyboardButton("📊 3. Получить сигнал", callback_data="get_signal")],
    ]
    markup = InlineKeyboardMarkup(keyboard)
    if update.callback_query:
        await update.callback_query.answer()
        await update.callback_query.message.edit_text(text, reply_markup=markup, parse_mode=ParseMode.HTML)
    else:
        await update.message.reply_text(text, reply_markup=markup, parse_mode=ParseMode.HTML)


async def back_to_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Выход из диалога ввода UID в главное меню."""
    await start(update, context)
    return ConversationHandler.END


# ---------- верификация ----------
async def request_uid(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    text = (
        "📝 <b>Верификация аккаунта</b>\n\n"
        f'1. Зарегистрируйтесь: <a href="{escape(REF_LINK)}">Pocket Option</a>\n'
        "2. Скопируйте ваш <b>UID</b> из личного кабинета.\n"
        "3. <b>Отправьте UID ответным сообщением.</b>\n\n"
        "⏳ После проверки администратором откроется доступ к сигналам."
    )
    kb = [[InlineKeyboardButton("⬅️ Назад", callback_data="main_menu")]]
    await query.message.edit_text(
        text, reply_markup=InlineKeyboardMarkup(kb), parse_mode=ParseMode.HTML, disable_web_page_preview=True
    )
    return AWAITING_UID


async def receive_uid(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_uid = update.message.text.strip()[:32]
    user = update.effective_user

    if not user_uid.isdigit():
        await update.message.reply_text("❗ UID — это только цифры. Отправьте его ещё раз.")
        return AWAITING_UID

    await update.message.reply_text(
        f"✅ <b>UID {escape(user_uid)} отправлен на проверку.</b>\n"
        "Мы пришлём уведомление, как только администратор подтвердит аккаунт.",
        parse_mode=ParseMode.HTML,
    )

    if ADMIN_ID:
        admin_text = (
            "🔔 <b>Новая заявка на верификацию</b>\n\n"
            f'👤 Пользователь: <a href="tg://user?id={user.id}">{escape(user.full_name)}</a>\n'
            f"🆔 Telegram ID: <code>{user.id}</code>\n"
            f"🔑 UID: <code>{escape(user_uid)}</code>"
        )
        kb = [
            [
                InlineKeyboardButton("✅ Одобрить", callback_data=f"approve_{user.id}"),
                InlineKeyboardButton("❌ Отклонить", callback_data=f"reject_{user.id}"),
            ]
        ]
        await context.bot.send_message(
            ADMIN_ID, admin_text, reply_markup=InlineKeyboardMarkup(kb), parse_mode=ParseMode.HTML
        )
    return ConversationHandler.END


async def admin_decision(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if query.from_user.id != ADMIN_ID:  # защита: решать может только админ
        await query.answer("Нет доступа", show_alert=True)
        return
    await query.answer()

    action, target = query.data.split("_", 1)
    target_id = int(target)
    base = query.message.text_html or ""

    if action == "approve":
        approve_user_id(target_id)
        await query.message.edit_text(f"{base}\n\n🟢 <b>СТАТУС: ОДОБРЕНО</b>", parse_mode=ParseMode.HTML)
        try:
            kb = [[InlineKeyboardButton("📊 Получить сигнал", callback_data="get_signal")]]
            await context.bot.send_message(
                target_id,
                "🎉 <b>Аккаунт верифицирован.</b> Сигналы доступны.",
                reply_markup=InlineKeyboardMarkup(kb),
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            logger.warning("Не удалось уведомить %s", target_id)
    else:
        await query.message.edit_text(f"{base}\n\n🔴 <b>СТАТУС: ОТКЛОНЕНО</b>", parse_mode=ParseMode.HTML)
        try:
            await context.bot.send_message(
                target_id,
                "❌ <b>UID не найден среди рефералов автора.</b>\n"
                "Убедитесь, что аккаунт создан по партнёрской ссылке, и отправьте UID повторно.",
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            logger.warning("Не удалось уведомить %s", target_id)


# ---------- сигналы ----------
async def get_signal_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if not is_allowed(query.from_user.id):
        text = (
            "🔒 <b>Доступ ограничен</b>\n\n"
            "Аккаунт ещё не верифицирован. Зарегистрируйтесь по ссылке и отправьте UID."
        )
        kb = [
            [InlineKeyboardButton("🔗 Регистрация Pocket Option", url=REF_LINK)],
            [InlineKeyboardButton("🆔 Подтвердить свой UID", callback_data="verify_uid")],
        ]
        await query.message.edit_text(text, reply_markup=InlineKeyboardMarkup(kb), parse_mode=ParseMode.HTML)
        return

    names = list(ASSETS.keys())
    keyboard = [
        [InlineKeyboardButton(n, callback_data=f"sig_{ASSETS[n]}") for n in names[i : i + 2]]
        for i in range(0, len(names), 2)
    ]
    keyboard.append([InlineKeyboardButton("⬅️ В главное меню", callback_data="main_menu")])
    await query.message.edit_text(
        "📊 <b>Выберите валютную пару:</b>",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode=ParseMode.HTML,
    )


async def process_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if not is_allowed(query.from_user.id):
        return

    ticker = query.data.replace("sig_", "", 1)
    asset_name = next((k for k, v in ASSETS.items() if v == ticker), None)
    if asset_name is None:
        return

    await query.message.edit_text(
        f"🔄 <b>Считаю сигнал для {asset_name}…</b>", parse_mode=ParseMode.HTML
    )
    direction, rsi, source = await asyncio.to_thread(generate_market_signal, ticker)

    minutes = random.choice([3, 5])
    exp_time = (datetime.now(TZ) + timedelta(minutes=minutes)).strftime("%H:%M")
    rsi_line = f"📉 RSI(14): <code>{rsi:.1f}</code>\n" if rsi is not None else ""

    text = (
        "🤖 <b>СИГНАЛ СФОРМИРОВАН</b>\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 Актив: <b>{asset_name}</b>\n"
        f"📈 Направление: {direction}\n"
        f"⏳ Экспирация: <b>{minutes} мин</b> (до {exp_time} МСК)\n"
        f"{rsi_line}"
        f"🧮 Основа: {source}\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"{DISCLAIMER}"
    )
    kb = [
        [InlineKeyboardButton("🔄 Другой актив", callback_data="get_signal")],
        [InlineKeyboardButton("🚀 Открыть Pocket Option", url=REF_LINK)],
    ]
    await query.message.edit_text(text, reply_markup=InlineKeyboardMarkup(kb), parse_mode=ParseMode.HTML)


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Действие отменено. Нажмите /start.")
    return ConversationHandler.END


def main():
    if not BOT_TOKEN:
        logger.error("Переменная BOT_TOKEN не установлена!")
        return

    app = Application.builder().token(BOT_TOKEN).build()

    conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(request_uid, pattern="^verify_uid$")],
        states={AWAITING_UID: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_uid)]},
        fallbacks=[
            CommandHandler("cancel", cancel),
            CallbackQueryHandler(back_to_menu, pattern="^main_menu$"),
        ],
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(conv)
    app.add_handler(CallbackQueryHandler(start, pattern="^main_menu$"))
    app.add_handler(CallbackQueryHandler(get_signal_menu, pattern="^get_signal$"))
    app.add_handler(CallbackQueryHandler(process_signal, pattern="^sig_"))
    app.add_handler(CallbackQueryHandler(admin_decision, pattern="^(approve|reject)_"))

    logger.info("Бот запущен")
    app.run_polling()


if __name__ == "__main__":
    main()
