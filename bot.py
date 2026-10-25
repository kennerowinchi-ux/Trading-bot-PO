import asyncio
import json
import logging
import math
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
CACHE_TTL = 45  # секунд

BOT_NAME = "Торговый бот Terret'a"

with open("pairs.json", encoding="utf-8") as f:
    PAIRS = json.load(f)  # {"EUR/USD": "EURUSD=X", ...}

# код -> (название, ориентир по экспирации, максимальный возраст последней свечи в минутах)
TIMEFRAMES = {
    "1m": ("M1", "1–3 мин", 10),
    "5m": ("M5", "5–10 мин", 25),
    "15m": ("M15", "15–30 мин", 50),
}
# горизонт симуляции в свечах
HORIZON_BARS = {"1m": 2, "5m": 2, "15m": 2}

LINE = "▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬"
THIN = "┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄"
TITLE = "🔱 <b>ТОРГОВЫЙ БОТ TERRET'A</b> 🔱"


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


# ======================= АНАЛИЗ 50/50 =======================
def phi(x: float) -> float:
    """CDF стандартного нормального распределения."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def shrink(p: float, k: float) -> float:
    """Сжатие оценки к 0.5: рынок на коротком горизонте почти не предсказуем."""
    return 0.5 + k * (p - 0.5)


def analyze_df(df: pd.DataFrame, horizon: int) -> dict:
    """
    «Аналитика 50/50».
    Считаем несколько статистических оценок вероятности роста цены, затем сжимаем
    их к 50% и усредняем. Итог почти всегда лежит в диапазоне 47–53%, то есть
    это по сути подбрасывание монеты с математическим оформлением.
    """
    close = df["Close"].astype(float)
    price = float(close.iloc[-1])
    r = np.log(close).diff().dropna().values
    eps = 1e-12

    # 1) Нормальная модель дрейфа (последние 30 свечей): P(r > 0) = Φ(μ/σ)
    n = 30
    rw = r[-n:]
    mu, sd = float(rw.mean()), float(rw.std(ddof=1)) + eps
    z_drift = mu / (sd / math.sqrt(n))
    p_norm = phi(mu / sd)

    # 2) Монте-Карло: бутстрэп доходностей последних 100 свечей, 5000 путей
    seed = int(df.index[-1].timestamp()) % (2**32)
    rng = np.random.default_rng(seed)
    sample = rng.choice(r[-100:], size=(5000, horizon), replace=True)
    p_mc = float((sample.sum(axis=1) > 0).mean())

    # 3) Возврат к среднему: z-score цены относительно SMA(20)
    sma = float(close.rolling(20).mean().iloc[-1])
    sstd = float(close.rolling(20).std().iloc[-1]) + eps
    z_mr = (price - sma) / sstd
    p_mr = 0.5 - 0.25 * math.tanh(z_mr / 2)  # выше среднего → чуть вероятнее откат вниз

    # 4) Моментум: доля положительных свечей из последних 10
    share_up = float((r[-10:] > 0).mean())
    p_mom = 0.5 + 0.5 * (share_up - 0.5)

    # Сжатие и усреднение
    parts = [
        ("Модель дрейфа Φ(μ/σ)", p_norm, 0.15),
        ("Монте-Карло, 5000 путей", p_mc, 0.20),
        ("Возврат к среднему (z-score)", p_mr, 0.20),
        ("Моментум последних 10 свечей", p_mom, 0.15),
    ]
    shrunk = [(name, raw, shrink(raw, k)) for name, raw, k in parts]
    p_up = float(np.mean([s for _, _, s in shrunk]))
    p_up = min(max(p_up, 0.45), 0.55)
    edge_pp = (p_up - 0.5) * 100  # отклонение от монетки в п.п.

    # Автокорреляция lag-1 и 95% «полоса шума»: если внутри — предсказуемости нет
    rr = r[-100:]
    rho1 = float(np.corrcoef(rr[:-1], rr[1:])[0, 1]) if len(rr) > 3 else 0.0
    if math.isnan(rho1):
        rho1 = 0.0
    noise_band = 1.96 / math.sqrt(len(rr))

    # Волатильность
    sd_short = float(r[-14:].std(ddof=1))
    sd_long = float(r[-100:].std(ddof=1)) + eps
    vr = sd_short / sd_long
    vol = "низкая" if vr < 0.8 else ("повышенная" if vr > 1.3 else "нормальная")

    if abs(edge_pp) < 0.5:
        direction = "COIN"
    else:
        direction = "UP" if edge_pp > 0 else "DOWN"

    return {
        "price": price,
        "p_up": p_up,
        "edge_pp": edge_pp,
        "direction": direction,
        "parts": shrunk,
        "z_drift": z_drift,
        "z_mr": z_mr,
        "rho1": rho1,
        "noise_band": noise_band,
        "sigma_pct": sd_short * 100,
        "volatility": vol,
        "horizon": horizon,
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
    if len(df) < 120:
        return None

    last = df.index[-1]
    last = last.tz_localize("UTC") if last.tzinfo is None else last.tz_convert("UTC")
    age_min = (pd.Timestamp.now(tz="UTC") - last).total_seconds() / 60

    result = analyze_df(df, HORIZON_BARS[interval])
    result["age_min"] = age_min
    result["stale"] = age_min > TIMEFRAMES[interval][2]
    _cache[key] = (time.time(), result)
    return result


def prob_bar(p_up: float, width: int = 20) -> str:
    filled = max(0, min(width, round(p_up * width)))
    return "🟩" * 0 + "▰" * filled + "▱" * (width - filled)


# ======================= ТЕКСТЫ =======================
WELCOME = (
    f"{TITLE}\n"
    f"{LINE}\n"
    "📈 <b>Статистический разбор</b> валютных пар Pocket Option\n"
    "🎲 Режим: <b>«50/50 + математика»</b>\n"
    "⏱ Таймфреймы: <b>M1 · M5 · M15</b>\n\n"
    "💠 <b>Тарифы доступа</b>\n"
    "  ┣ 🆓 Базовый — <b>10 анализов в сутки</b>\n"
    "  ┗ 💎 VIP — <b>без ограничений</b>\n\n"
    "<blockquote>⚠️ <b>Честно:</b> это <u>неправильный и не прогностический</u> анализ. "
    "Математика здесь — оформление: реальная вероятность на коротком горизонте ≈ <b>50%</b>, "
    "как у монетки. Не инвестиционная рекомендация.</blockquote>\n"
    f"{LINE}\n"
    "👇 <b>Выберите раздел:</b>"
)

METHOD = (
    "<b>📚 КАК ЭТО СЧИТАЕТСЯ</b>\n"
    f"{LINE}\n"
    "<b>1️⃣ Данные.</b> Свечи M1 / M5 / M15 с Yahoo Finance. Цены могут отличаться от "
    "Pocket Option; в выходные рынок закрыт и анализ не выполняется.\n\n"
    "<b>2️⃣ Четыре статистические оценки</b> вероятности роста цены:\n"
    "  ◆ <b>Дрейф Φ(μ/σ)</b> — нормальная модель по 30 свечам\n"
    "  ◆ <b>Монте-Карло</b> — 5000 бутстрэп-путей из реальных доходностей\n"
    "  ◆ <b>Z-score к SMA(20)</b> — склонность к возврату к среднему\n"
    "  ◆ <b>Моментум</b> — доля растущих свечей из последних 10\n\n"
    "<b>3️⃣ Сжатие к 50%.</b> Каждая оценка притягивается к 0.5 "
    "<code>p = 0.5 + k·(p₀ − 0.5)</code>, так как на коротком горизонте цена близка "
    "к случайному блужданию. Затем оценки усредняются, итог ограничен диапазоном 45–55%.\n\n"
    "<b>4️⃣ Проверка на шум.</b> Считается автокорреляция доходностей lag-1 и "
    "95%-полоса шума ±1.96/√n. Почти всегда значение внутри полосы — предсказуемости нет.\n\n"
    "<b>5️⃣ Результат.</b> «Вверх» или «Вниз» — по знаку отклонения от 50%. "
    "Если оно меньше 0.5 п.п. — бот прямо пишет «монетка».\n\n"
    "<blockquote>❗ <b>Это неправильный анализ.</b> Процент — не вероятность выигрыша и не "
    "гарантия. Отклонение в 1–3 п.п. — статистический шум; с учётом выплат брокера "
    "ожидание по такой торговле отрицательное.</blockquote>\n"
    f"{LINE}"
)

RISKS = (
    "<b>⚠️ РАСКРЫТИЕ РИСКОВ</b>\n"
    f"{LINE}\n"
    "🔸 Бинарные опционы — высокий риск потери всех вложенных средств. "
    "Большинство трейдеров несёт убытки.\n"
    "🔸 Анализ в этом боте <b>неправильный и не имеет доказанной предсказательной силы</b>.\n"
    "🔸 Ни один индикатор и ни одна модель не гарантируют прибыль.\n"
    "🔸 Торгуйте только теми средствами, потерю которых можете себе позволить.\n"
    "🔸 Материалы бота — информационные, не инвестиционная рекомендация.\n"
    "🔸 Ссылка на регистрацию — партнёрская ссылка автора: он может получать "
    "вознаграждение от брокера.\n"
    f"{LINE}"
)

SHORT_DISCLAIMER = (
    "<blockquote>⚠️ <b>Неправильный анализ.</b> Реальная вероятность ≈ 50/50. "
    "Не инвестиционная рекомендация, результат не гарантирован.</blockquote>"
)


# ======================= КЛАВИАТУРЫ =======================
def main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("📊 ✦ Новый анализ ✦", callback_data="pairs")],
            [
                InlineKeyboardButton("📚 Методика", callback_data="method"),
                InlineKeyboardButton("⚠️ Риски", callback_data="risks"),
            ],
            [
                InlineKeyboardButton("👤 Профиль", callback_data="status"),
                InlineKeyboardButton("🔗 Регистрация", callback_data="ref"),
            ],
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
    f"{TITLE}\n"
    f"{LINE}\n"
    "🔒 <b>Доступ ограничен</b>\n\n"
    "Для работы нужна подписка на канал автора.\n\n"
    "  ┣ 1️⃣ Подпишитесь на канал\n"
    "  ┗ 2️⃣ Нажмите «Проверить подписку»\n\n"
    "🆓 Основной канал — <b>10 анализов в сутки</b>\n"
    "💎 VIP-канал — <b>без ограничений</b>"
)


async def safe_edit(q, text: str, kb: InlineKeyboardMarkup | None = None):
    try:
        await q.edit_message_text(
            text, parse_mode=ParseMode.HTML, reply_markup=kb, disable_web_page_preview=True
        )
    except BadRequest as e:
        if "not modified" not in str(e).lower():
            raise


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
            "<b>🪪 Подтверждение регистрации</b>\n\n"
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
                f"Подтвердить: <code>/approve {user.id}</code>",
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
        await context.bot.send_message(uid, "✅ <b>Регистрация подтверждена.</b> Спасибо!", parse_mode=ParseMode.HTML)
    except Exception as e:
        log.warning("Не удалось уведомить пользователя: %s", e)


# ======================= КНОПКИ =======================
def build_result_text(pair: str, tf_name: str, expiry: str, res: dict, remaining: str) -> str:
    decimals = 3 if "JPY" in pair else 5
    p_up = res["p_up"]
    p_down = 1 - p_up
    edge = res["edge_pp"]

    if res["direction"] == "UP":
        head = "🟢 <b>▲ ВВЕРХ (CALL)</b>"
    elif res["direction"] == "DOWN":
        head = "🔴 <b>▼ ВНИЗ (PUT)</b>"
    else:
        head = "🪙 <b>МОНЕТКА</b> — перевеса нет"

    ae = abs(edge)
    if ae < 0.5:
        edge_word = "практически ноль"
    elif ae < 1.5:
        edge_word = "ничтожный"
    else:
        edge_word = "небольшой"

    parts_txt = "\n".join(
        f"  ◆ {escape(name)}\n      <code>{raw * 100:5.1f}%</code> → <code>{sh * 100:5.2f}%</code>"
        for name, raw, sh in res["parts"]
    )

    rho_state = "внутри полосы шума" if abs(res["rho1"]) <= res["noise_band"] else "вне полосы (вероятно, случайность)"

    now = datetime.now(TZ).strftime("%d.%m.%Y %H:%M МСК")
    return (
        f"{TITLE}\n"
        f"<b>📊 {pair} · {tf_name}</b>\n"
        f"🕒 <code>{now}</code>\n"
        f"{LINE}\n"
        f"🎯 <b>РЕЗУЛЬТАТ «50/50»</b>\n"
        f"{head}\n\n"
        f"▲ <b>{p_up * 100:.1f}%</b>  {prob_bar(p_up)}  <b>{p_down * 100:.1f}%</b> ▼\n"
        f"Отклонение от монетки: <b>{edge:+.2f} п.п.</b> ({edge_word})\n"
        f"{THIN}\n"
        f"🧮 <b>МАТЕМАТИКА</b> <i>(сырая → после сжатия)</i>\n{parts_txt}\n"
        f"{THIN}\n"
        f"🔬 <b>ПРОВЕРКА НА ШУМ</b>\n"
        f"  ┣ z-дрейф: <code>{res['z_drift']:+.2f}</code>\n"
        f"  ┣ z к SMA(20): <code>{res['z_mr']:+.2f}</code>\n"
        f"  ┗ автокорр. lag-1: <code>{res['rho1']:+.3f}</code> (±{res['noise_band']:.3f}) — {rho_state}\n"
        f"{LINE}\n"
        f"💹 <b>РЫНОК</b>\n"
        f"  ┣ Цена: <code>{res['price']:.{decimals}f}</code>\n"
        f"  ┣ Волатильность: {res['volatility']} (σ≈{res['sigma_pct']:.3f}%)\n"
        f"  ┗ Ориентир по экспирации: {expiry}\n"
        f"{LINE}\n"
        f"{remaining}\n\n"
        f"{SHORT_DISCLAIMER}"
    )


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

    elif action == "risks":
        await safe_edit(q, RISKS, InlineKeyboardMarkup([back_row()]))

    elif action == "ref":
        text = (
            "<b>🔗 РЕГИСТРАЦИЯ НА POCKET OPTION</b>\n"
            f"{LINE}\n"
            "Это партнёрская ссылка автора бота: при регистрации по ней автор может "
            "получать вознаграждение от брокера. Для вас условия не меняются.\n\n"
            "<b>Порядок действий</b>\n"
            "  1️⃣ Откройте ссылку и создайте аккаунт\n"
            "  2️⃣ Скопируйте свой ID из профиля Pocket Option\n"
            "  3️⃣ Отправьте боту: <code>/uid 12345678</code>\n"
            "  4️⃣ Дождитесь подтверждения в разделе «Профиль»\n\n"
            "<blockquote>Начните с демо-счёта и не вкладывайте средства, "
            "которые не готовы потерять.</blockquote>"
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
            "<b>👤 ПРОФИЛЬ</b>\n"
            f"{LINE}\n"
            f"  ┣ ID Telegram: <code>{user.id}</code>\n"
            f"  ┣ Тариф: <b>{plan}</b>\n"
            f"  ┣ Анализы: {limit}\n"
            f"  ┗ Регистрация: {reg}\n"
            f"{LINE}\n"
            "<i>Счётчик обновляется ежедневно в 00:00 МСК.</i>"
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
            f"{TITLE}\n"
            "<b>📊 НОВЫЙ АНАЛИЗ</b>\n"
            f"{LINE}\n"
            "Шаг <b>1</b> из <b>2</b> — выберите валютную пару:",
            InlineKeyboardMarkup(rows),
        )

    elif action.startswith("pair:"):
        pair = action[5:]
        if pair not in PAIRS:
            return
        row = [
            InlineKeyboardButton(f"⏱ {name} · экспирация {exp}", callback_data=f"sig:{pair}:{code}")
            for code, (name, exp, _) in TIMEFRAMES.items()
        ]
        kb = [[b] for b in row] + [[InlineKeyboardButton("⬅️ К выбору пары", callback_data="pairs")]]
        await safe_edit(
            q,
            f"{TITLE}\n"
            f"<b>📊 {pair}</b>\n"
            f"{LINE}\n"
            "Шаг <b>2</b> из <b>2</b> — выберите таймфрейм.\n\n"
            "<blockquote>Чем меньше таймфрейм, тем больше рыночного шума.</blockquote>",
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
                "<b>⛔ ДНЕВНОЙ ЛИМИТ ИСЧЕРПАН</b>\n"
                f"{LINE}\n"
                f"Вы использовали <b>{DAILY_LIMIT} из {DAILY_LIMIT}</b> анализов.\n"
                "Лимит обновится в 00:00 МСК.\n\n"
                "Для работы без ограничений подключите VIP-доступ.",
                InlineKeyboardMarkup(
                    [
                        [InlineKeyboardButton("💎 VIP-канал", url=VIP_CHANNEL_URL)],
                        [InlineKeyboardButton("💬 Связаться с автором", url=f"https://t.me/{CONTACT_USERNAME}")],
                        back_row(),
                    ]
                ),
            )
            return

        tf_name, expiry, _ = TIMEFRAMES[tf]
        await safe_edit(q, f"⏳ <b>{pair} · {tf_name}</b>\n🧮 Загружаю котировки и запускаю симуляцию…")

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

        if res["stale"]:
            await safe_edit(
                q,
                "<b>🕒 РЫНОК СЕЙЧАС ЗАКРЫТ</b>\n"
                f"{LINE}\n"
                f"Последняя доступная котировка по {pair} получена {res['age_min'] / 60:.1f} ч назад. "
                "Анализ по устаревшим данным не выполняется.\n\n"
                "Форекс торгуется с понедельника по пятницу. Анализ не засчитан в лимит.",
                back_kb,
            )
            return

        rec = data["usage"].get(str(user.id))
        if not rec or rec["date"] != today():
            rec = {"date": today(), "count": 0}
        rec["count"] += 1
        data["usage"][str(user.id)] = rec
        save_data(data)

        if tier == "vip":
            remaining = "💎 <b>VIP</b> — без ограничений"
        else:
            remaining = f"🆓 Осталось анализов сегодня: <b>{DAILY_LIMIT - rec['count']}</b> из {DAILY_LIMIT}"

        text = build_result_text(pair, tf_name, expiry, res, remaining)
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
        await safe_edit(q, text, kb)


def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(CommandHandler("uid", uid_cmd))
    app.add_handler(CommandHandler("approve", approve_cmd))
    app.add_handler(CallbackQueryHandler(on_button))
    app.run_polling()


if __name__ == "__main__":
    main()
