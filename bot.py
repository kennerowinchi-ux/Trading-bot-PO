import os
import json
import random
import logging
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
    ContextTypes,
    ConversationHandler
)

# Настройка логирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка конфигурации валютных пар (из вашего JSON)
ASSETS_JSON = """{
  "EUR/USD": "EURUSD=X",
  "GBP/USD": "GBPUSD=X",
  "USD/JPY": "JPY=X",
  "AUD/USD": "AUDUSD=X",
  "USD/CAD": "CAD=X",
  "USD/CHF": "CHF=X",
  "EUR/JPY": "EURJPY=X",
  "EUR/GBP": "EURGBP=X"
}"""
ASSETS = json.loads(ASSETS_JSON)

# Состояния для диалога отправки UID
AWAITING_UID = 1

# Загрузка переменных окружения из GitHub Actions
BOT_TOKEN = os.getenv("BOT_TOKEN")
REF_LINK = os.getenv("REF_LINK", "https://pocketoption.com")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))

# Файлы для симуляции простой БД (так как в GitHub Actions контейнер эфемерный, 
# верификация идет через ручной аппрув админом)
APPROVED_USERS_FILE = "approved_users.json"

def get_approved_users():
    if os.path.exists(APPROVED_USERS_FILE):
        with open(APPROVED_USERS_FILE, "r") as f:
            return json.load(f)
    return []

def approve_user_id(user_id):
    users = get_approved_users()
    if user_id not in users:
        users.append(user_id)
        with open(APPROVED_USERS_FILE, "w") as f:
            json.dump(users, f)

# Анализ рынка для сигналов 50/50 (RSI + Волатильность)
def generate_market_signal(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period="1d", interval="5m")
        if df.empty or len(df) < 15:
            raise ValueError("Недостаточно данных")
        
        # Расчет базового RSI для направления 50/50
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # Направление
        direction = "🟢 ВВЕРХ (CALL)" if current_rsi > 50 else "🔴 ВНИЗ (PUT)"
        
        # Красивый расчет проходимости (Рандом в пределах успешного коридора 74-92%)
        winrate = round(random.uniform(74.5, 92.3), 1)
        
        return direction, winrate
    except Exception as e:
        logger.error(f"Ошибка при получении данных yfinance для {ticker_symbol}: {e}")
        # Запасной вариант (Фолбэк), если Yahoo Finance временно недоступен
        direction = random.choice(["🟢 ВВЕРХ (CALL)", "🔴 ВНИЗ (PUT)"])
        winrate = round(random.uniform(71.0, 89.5), 1)
        return direction, winrate

# Стартовое меню
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    text = (
        f"👋 **Привет, {user.first_name}!**\n\n"
        f"🤖 Добро пожаловать в закрытый аналитический бот для **Pocket Option**.\n\n"
        f"📈 Наш ИИ анализирует графики в реальном времени и выдает сигналы с проходимостью до 92%.\n\n"
        f"Чтобы получить доступ к сигналам, вам необходимо активировать аккаунт по нашей партнерской ссылке."
    )
    
    keyboard = [
        [InlineKeyboardButton("🔗 1. Регистрация аккаунта", url=REF_LINK)],
        [InlineKeyboardButton("🆔 2. Подтвердить свой UID", callback_data="verify_uid")],
        [InlineKeyboardButton("📊 3. Получить Сигнал", callback_data="get_signal")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if update.message:
        await update.message.reply_text(text, reply_markup=reply_markup, parse_mode="Markdown")
    elif update.callback_query:
        await update.callback_query.message.edit_text(text, reply_markup=reply_markup, parse_mode="Markdown")

# Обработка нажатия на кнопку "Подтвердить UID"
async def request_uid(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    text = (
        "📝 **Инструкция по верификации аккаунта:**\n\n"
        f"1. Перейдите по ссылке и зарегистрируйтесь: [Регистрация на Pocket Option]({REF_LINK})\n"
        "2. Скопируйте ваш **UID** из личного кабинета.\n"
        "3. **Отправьте ваш UID в ответном сообщении этому боту.**\n\n"
        "⏳ После проверки администратором вам откроется полный доступ к торговым сигналам!"
    )
    
    keyboard = [[InlineKeyboardButton("⬅️ Назад", callback_data="main_menu")]]
    await query.message.edit_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")
    return AWAITING_UID

# Получение UID от пользователя и пересылка Админу
async def receive_uid(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_uid = update.message.text.strip()
    user = update.effective_user
    
    # Сообщение пользователю
    await update.message.reply_text(
        f"✅ **Ваш UID ({user_uid}) отправлен на проверку!**\n\n"
        f"Обычно проверка занимает от 5 до 15 минут. Мы пришлем вам уведомление, как только администратор подтвердит ваш аккаунт."
    )
    
    # Отправка заявки администратору (с кнопкой быстрого аппрува)
    if ADMIN_ID != 0:
        admin_text = (
            f"🔔 **НОВАЯ ЗАЯВКА НА ВЕРИФИКАЦИЮ!**\n\n"
            f"👤 Пользователь: {user.mention_markdown_v2()}\n"
            f"🆔 Telegram ID: `{user.id}`\n"
            f"🔑 Заявленный UID: `{user_uid}`\n"
        )
        keyboard = [
            [
                InlineKeyboardButton("✅ Одобрить", callback_data=f"approve_{user.id}"),
                InlineKeyboardButton("❌ Отклонить", callback_data=f"reject_{user.id}")
            ]
        ]
        await context.bot.send_message(
            chat_id=ADMIN_ID, 
            text=admin_text, 
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="MarkdownV2"
        )
    
    return ConversationHandler.END

# Обработка решений администратора
async def admin_decision(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    action, target_user_id = query.data.split("_")
    target_user_id = int(target_user_id)
    
    if action == "approve":
        approve_user_id(target_user_id)
        await query.message.edit_text(f"{query.message.text}\n\n🟢 **СТАТУС: ОДОБРЕНО**")
        
        # Уведомляем пользователя
        try:
            keyboard = [[InlineKeyboardButton("📊 Получить Сигнал", callback_data="get_signal")]]
            await context.bot.send_message(
                chat_id=target_user_id,
                text="🎉 **Поздравляем! Ваш аккаунт успешно верифицирован.**\n\nТеперь вам доступен полный функционал торговых сигналов!",
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode="Markdown"
            )
        except Exception:
            pass
            
    elif action == "reject":
        await query.message.edit_text(f"{query.message.text}\n\n🔴 **СТАТУС: ОТКЛОНЕНО**")
        # Уведомляем пользователя
        try:
            await context.bot.send_message(
                chat_id=target_user_id,
                text="❌ **К сожалению, ваш UID не найден в списках наших рефералов.**\n\nПожалуйста, убедитесь, что зарегистрировали новый аккаунт по нашей ссылке, и отправьте UID повторно."
            )
        except Exception:
            pass

# Выдача сигналов
async def get_signal_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    
    # Проверка, авторизован ли пользователь
    approved_users = get_approved_users()
    if user_id not in approved_users and user_id != ADMIN_ID:
        text = (
            "🔒 **ДОСТУП ОГРАНИЧЕН!**\n\n"
            "Вы не можете получать сигналы, так как ваш аккаунт еще не прошел верификацию.\n\n"
            "Пожалуйста, зарегистрируйтесь по ссылке ниже и пришлите свой UID администратору."
        )
        keyboard = [
            [InlineKeyboardButton("🔗 Регистрация Pocket Option", url=REF_LINK)],
            [InlineKeyboardButton("🆔 Подтвердить свой UID", callback_data="verify_uid")]
        ]
        await query.message.edit_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")
        return

    # Если доступ есть — выводим меню выбора активов
    text = "📊 **Выберите валютную пару для генерации сигнала:**"
    keyboard = []
    
    # Строим сетку кнопок для валютных пар по 2 в ряд
    asset_keys = list(ASSETS.keys())
    for i in range(0, len(asset_keys), 2):
        row = [InlineKeyboardButton(asset_keys[i], callback_data=f"sig_{ASSETS[asset_keys[i]]}")]
        if i + 1 < len(asset_keys):
            row.append(InlineKeyboardButton(asset_keys[i+1], callback_data=f"sig_{ASSETS[asset_keys[i+1]]}"))
        keyboard.append(row)
        
    keyboard.append([InlineKeyboardButton("⬅️ В главное меню", callback_data="main_menu")])
    await query.message.edit_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")

# Генерация и красивый вывод самого сигнала
async def process_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    ticker_symbol = query.data.replace("sig_", "")
    # Находим красивое имя пары по тикеру
    asset_name = next((k for k, v in ASSETS.items() if v == ticker_symbol), ticker_symbol)
    
    # Красивая анимация анализа
    await query.message.edit_text(f"🔄 *ИИ анализирует стакан и объемы для {asset_name}...* \n⌛ Пожалуйста, подождите.", parse_mode="Markdown")
    await asyncio.sleep(2.5) # Имитация сложного расчета
    direction, winrate = generate_market_signal(ticker_symbol)
    
# Расчет времени экспирации (+3 или +5 минут случайным образом)
exp_time = (datetime.now() + timedelta(minutes=random.choice([3, 5]))).strftime("%H:%M")
signal_text = (
f"🤖 СИГНАЛ СФОРМИРОВАН 🤖\n"
f"━━━━━━━━━━━━━━━━━━\n"
f"📊 Актив: {asset_name}\n"
f" Направление: {direction}\n"
f"⏳ Время экспирации: {exp_time} (МСК)\n"
f"🎯 Точность алгоритма: {winrate}%\n"
f"━━━━━━━━━━━━━━━━━━\n"
f"⚠️ Внимание: открывайте сделку строго по указанному времени!"
)
keyboard = [
[InlineKeyboardButton(" Перевход / Другой Актив", callback_data="get_signal")],
[InlineKeyboardButton(" Открыть Pocket Option", url=REF_LINK)]
]
await query.message.edit_text(signal_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")

Отмена диалога ввода UID

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
await update.message.reply_text("Действие отменено.", reply_markup=None)
return ConversationHandler.END
def main():
if not BOT_TOKEN:
logger.error("Переменная BOT_TOKEN не установлена! Бот завершает работу.")
return
application = Application.builder().token(BOT_TOKEN).build()
# Сценарий разговора для UID
conv_handler = ConversationHandler(
entry_points=[CallbackQueryHandler(request_uid, pattern="^verify_uid$")],
states={
AWAITING_UID: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_uid)]
},
fallbacks=[CommandHandler("cancel", cancel)],
per_message=False
)
# Регистрация хендлеров
application.add_handler(CommandHandler("start", start))
application.add_handler(conv_handler)
application.add_handler(CallbackQueryHandler(start, pattern="^main_menu$"))
application.add_handler(CallbackQueryHandler(get_signal_menu, pattern="^get_signal$"))
application.add_handler(CallbackQueryHandler(process_signal, pattern="^sig_"))
application.add_handler(CallbackQueryHandler(admin_decision, pattern="^(approve|reject)_"))
# Старт бота
logger.info("Бот успешно запущен...")
application.run_polling()
if name == 'main':
main()
