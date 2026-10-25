import asyncio
import json
import logging
import math
import os
import time
import zlib
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
    MessageHandler,
    filters,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bot")

# ======================= НАСТРОЙКИ (переменные окружения / секреты) =======================
BOT_TOKEN = os.environ["BOT_TOKEN"]
FREE_CHANNEL_ID = int(os.environ["FREE_CHANNEL_ID"])
VIP_CHANNEL_ID = int(os.environ["VIP_CHANNEL_ID"])
FREE_CHANNEL_URL = os.getenv("FREE_CHANNEL_URL", "https://t.me/+F_OldsGjeK5lMjBi")
VIP_CHANNEL_URL = os.getenv("VIP_CHANNEL_URL", "https://t.me/+itYuwMLRnBcyNDEy")
REF_LINK = os.environ["REF_LINK"]  # твоя реферальная ссылка Pocket Option
ADMIN_ID = int(os.getenv("ADMIN_ID") or 0)  # твой Telegram ID: сюда приходят заявки
CONTACT_USERNAME = os.getenv("CONTACT_USERNAME", "Terret911")  # поддержка
TZ = ZoneInfo("Europe/Moscow")
DATA_FILE = "data.json"
CACHE_TTL = 45  # секунд

BOT_NAME = "Торговый бот Terret'a"

with open("pairs.json", encoding="utf-8") as f:
    PAIRS = json.load(f)  # {"EUR/USD": "EURUSD=X", ...}

# код -> (название, ориентир по экспирации, макс. возраст последней свечи в минутах, горизонт в свечах)
TIMEFRAMES = {
    "1m": ("M1", "1–3 мин", 10, 2),
    "5m": ("M5", "5–10 мин", 25, 2),
    "15m": ("M15", "15–30 мин", 50, 2),
}

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
    data.setdefault("users", {})  # user_id -> {"name", "po_id", "status": pending|approved|rejected}
    return data


def save_data(data: dict) -> None:
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def status_of(data: dict, user_id: int) -> str:
    if user_id == ADMIN_ID:
        return "approved"
    return data["users"].get(str(user_id), {}).get("status", "new")


# ======================= ПОДПИСКА НА КАНАЛЫ =======================
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
    if user_id == ADMIN_ID:
        return "vip"
    if await is_member(bot, VIP_CHANNEL_ID, user_id):
        return "vip"
    if await is_member(bot, FREE_CHANNEL_ID, user_id):
        return "free"
    return "none"


LOCKED = (
    f"{TITLE}\n"
    f"{LINE}\n"
    "🔒 <b>Доступ ограничен</b>\n\n"
    "Для работы нужна подписка на канал автора.\n\n"
    "  ┣ 1️⃣ Подпишитесь на канал\n"
    "  ┗ 2️⃣ Нажмите «Проверить подписку»\n\n"
    "🆓 Основной канал · 💎 VIP-канал"
)


def subscribe_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("📢 Основной канал", url=FREE_CHANNEL_URL)],
            [InlineKeyboardButton("💎 VIP-канал", url=VIP_CHANNEL_URL)],
            [InlineKeyboardButton("✅ Проверить подписку", callback_data="menu")],
        ]
    )


# ======================= АНАЛИЗ 50/50 =======================
def phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def shrink(p: float, k: float) -> float:
    return 0.5 + k * (p - 0.5)


def analyze(close: np.ndarray, horizon: int, seed: int) -> dict:
    """
    «Аналитика 50/50»: несколько статистических оценок вероятности роста,
    сжатие к 50% и усреднение. Итог лежит в диапазоне 45–55%.
    """
    close = np.asarray(close, dtype=float)
    r = np.diff(np.log(close))
    eps = 1e-12

    n = 30
    rw = r[-n:]
    mu, sd = float(rw.mean()), float(rw.std(ddof=1)) + eps
    z_drift = mu / (sd / math.sqrt(n))
    p_norm = phi(mu / sd)

    rng = np.random.default_rng(seed)
    sample = rng.choice(r[-100:], size=(5000, horizon), replace=True)
    p_mc = float((sample.sum(axis=1) > 0).mean())

    sma = float(close[-20:].mean())
    sstd = float(close[-20:].std(ddof=1)) + eps
    z_mr = (float(close[-1]) - sma) / sstd
    p_mr = 0.5 - 0.25 * math.tanh(z_mr / 2)

    share_up = float((r[-10:] > 0).mean())
    p_mom = 0.5 + 0.5 * (share_up - 0.5)

    parts = [
        ("Модель дрейфа Φ(μ/σ)", p_norm, 0.15),
        ("Монте-Карло, 5000 путей", p_mc, 0.20),
        ("Возврат к среднему (z-score)", p_mr, 0.20),
        ("Моментум последних 10 свечей", p_mom, 0.15),
    ]
    shrunk = [(name, raw, shrink(raw, k)) for name, raw, k in parts]
    p_up = min(max(float(np.mean([s for _, _, s in shrunk])), 0.45), 0.55)
    edge_pp = (p_up - 0.5) * 100

    rr = r[-100:]
    rho1 = float(np.corrcoef(rr[:-1], rr[1:])[0, 1]) if len(rr) > 3 else 0.0
    if math.isnan(rho1):
        rho1 = 0.0
    noise_band = 1.96 / math.sqrt(len(rr))

    sd_short = float(r[-14:].std(ddof=1))
    sd_long = float(r[-100:].std(ddof=1)) + eps
    vr = sd_short / sd_long
    vol = "низкая" if vr < 0.8 else ("повышенная" if vr > 1.3 else "нормальная")

    direction = "UP" if edge_pp >= 0 else "DOWN"

    return {
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
    }


_cache: dict = {}


def fetch_live(ticker: str, interval: str) -> dict | None:
    """Реальные котировки Yahoo Finance. None — если данных нет или рынок закрыт."""
    key = (ticker, interval)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]

    df = yf.download(ticker, period="5d", interval=interval, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Close"])
    if len(df) < 120:
        return None

    last = df.index[-1]
    last = last.tz_localize("UTC") if last.tzinfo is None else last.tz_convert("UTC")
    age_min = (pd.Timestamp.now(tz="UTC") - last).total_seconds() / 60
    if age_min > TIMEFRAMES[interval][2]:
        return None  # рынок закрыт — перейдём в OTC-режим

    close = df["Close"].astype(float).values
    res = analyze(close, TIMEFRAMES[interval][3], int(last.timestamp()) % (2**32))
    res["price"] = float(close[-1])
    res["source"] = "live"
    _cache[key] = (time.time(), res)
    return res


def make_otc(pair: str, interval: str) -> dict:
    """OTC-режим: рынок закрыт, расчёт по смоделированному ряду (без реальных котировок)."""
    seed = zlib.crc32(f"{pair}|{interval}|{int(time.time() // 60)}".encode()) % (2**32)
    rng = np.random.default_rng(seed)
    close = np.exp(np.cumsum(rng.normal(0, 1e-4, 200)))
    res = analyze(close, TIMEFRAMES[interval][3], seed)
    res["price"] = None
    res["source"] = "otc"
    return res


def get_signal(pair: str, interval: str) -> dict:
    try:
        res = fetch_live(PAIRS[pair], interval)
    except Exception as e:
        log.warning("live fetch failed: %s", e)
        res = None
    return res or make_otc(pair, interval)


def prob_bar(p_up: float, width: int = 20) -> str:
    filled = max(0, min(width, round(p_up * width)))
    return "▰" * filled + "▱" * (width - filled)


# ======================= ТЕКСТЫ =======================
def welcome_text() -> str:
    return (
        f"{TITLE}\n"
        f"{LINE}\n"
        "📈 <b>Математический разбор</b> валютных пар <b>Pocket Option</b>\n"
        "🎲 Режим: <b>«50/50»</b> · круглосуточно, 7 дней в неделю\n"
        "⏱ Таймфреймы: <b>M1 · M5 · M15</b>\n\n"
        "<blockquote>⚠️ Это упрощённая статистическая оценка, а не гарантия. "
        "Реальная вероятность на короткой экспирации ≈ 50%. "
        "Не инвестиционная рекомендация.</blockquote>\n"
        f"{LINE}\n"
        "👇 <b>Выберите раздел:</b>"
    )


REGISTER_TEXT = (
    f"{TITLE}\n"
    f"{LINE}\n"
    "🔐 <b>РЕГИСТРАЦИЯ</b>\n\n"
    "Чтобы получить доступ к сигналам, нужно зарегистрироваться на "
    "<b>Pocket Option</b> по ссылке автора:\n\n"
    "  1️⃣ Нажмите «Открыть Pocket Option» и создайте аккаунт\n"
    "  2️⃣ Скопируйте свой <b>ID</b> из профиля Pocket Option\n"
    "  3️⃣ Нажмите «Отправить ID» и пришлите его сообщением\n"
    "  4️⃣ Дождитесь подтверждения — придёт уведомление\n\n"
    "<blockquote>Это партнёрская ссылка автора бота: он может получать "
    "вознаграждение от брокера. Для вас условия не меняются.</blockquote>"
)

RISKS = (
    "<b>⚠️ РАСКРЫТИЕ РИСКОВ</b>\n"
    f"{LINE}\n"
    "🔸 Бинарные опционы — высокий риск потери всех вложенных средств. "
    "Большинство трейдеров несёт убытки.\n"
    "🔸 Оценка в боте упрощённая и <b>не имеет доказанной предсказательной силы</b>: "
    "на короткой экспирации исход близок к 50/50.\n"
    "🔸 В выходные и нерабочие часы бот работает в OTC-режиме — по смоделированным данным, "
    "без реальных котировок.\n"
    "🔸 Ни одна модель не гарантирует прибыль.\n"
    "🔸 Торгуйте только теми средствами, потерю которых можете себе позволить. "
    "Начните с демо-счёта.\n"
    "🔸 Ссылка на регистрацию — партнёрская ссылка автора: он может получать "
    "вознаграждение от брокера.\n"
    f"{LINE}"
)

SHORT_DISCLAIMER = (
    "<blockquote>⚠️ Упрощённая оценка. Реальная вероятность ≈ 50/50. "
    "Не инвестиционная рекомендация, результат не гарантирован.</blockquote>"
)


# ======================= КЛАВИАТУРЫ =======================
def main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("📊 ✦ Новый сигнал ✦", callback_data="pairs")],
            [
                InlineKeyboardButton("👤 Профиль", callback_data="status"),
                InlineKeyboardButton("⚠️ Риски", callback_data="risks"),
            ],
            [InlineKeyboardButton("💬 Поддержка", url=f"https://t.me/{CONTACT_USERNAME}")],
        ]
    )


def back_row():
    return [InlineKeyboardButton("⬅️ Главное меню", callback_data="menu")]


def register_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🚀 Открыть Pocket Option", url=REF_LINK)],
            [InlineKeyboardButton("🪪 Отправить ID", callback_data="send_id")],
            [InlineKeyboardButton("💬 Поддержка", url=f"https://t.me/{CONTACT_USERNAME}")],
        ]
    )


async def safe_edit(q, text: str, kb: InlineKeyboardMarkup | None = None):
    try:
        await q.edit_message_text(
            text, parse_mode=ParseMode.HTML, reply_markup=kb, disable_web_page_preview=True
        )
    except BadRequest as e:
        if "not modified" not in str(e).lower():
            raise


def gate_text(status: str) -> str:
    if status == "pending":
        return (
            f"{TITLE}\n{LINE}\n"
            "⏳ <b>Заявка на проверке</b>\n\n"
            "Ваш ID получен и отправлен автору. Как только регистрацию подтвердят — "
            "придёт уведомление и откроется доступ."
        )
    if status == "rejected":
        return (
            f"{TITLE}\n{LINE}\n"
            "❌ <b>Заявка отклонена</b>\n\n"
            "Регистрацию по ссылке автора подтвердить не удалось. "
            "Проверьте, что аккаунт создан по ссылке ниже, и отправьте ID снова — "
            "или напишите в поддержку."
        )
    return REGISTER_TEXT


# ======================= РЕГИСТРАЦИЯ =======================
async def submit_uid(update: Update, context: ContextTypes.DEFAULT_TYPE, po_id: str):
    user = update.effective_user
    po_id = po_id.strip()[:32]
    if not po_id.isdigit():
        await update.message.reply_text(
            "❗ ID должен состоять только из цифр. Скопируйте его из профиля Pocket Option "
            "и отправьте ещё раз."
        )
        context.user_data["await_uid"] = True
        return

    data = load_data()
    data["users"][str(user.id)] = {
        "name": user.username or user.full_name or str(user.id),
        "po_id": po_id,
        "status": "pending",
    }
    save_data(data)
    context.user_data["await_uid"] = False

    await update.message.reply_text(
        "✅ <b>ID получен.</b>\nЗаявка отправлена автору. Вы получите уведомление, "
        "как только её подтвердят.",
        parse_mode=ParseMode.HTML,
    )
    if ADMIN_ID:
        kb = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("✅ Подтвердить", callback_data=f"adm:ok:{user.id}"),
                    InlineKeyboardButton("❌ Отклонить", callback_data=f"adm:no:{user.id}"),
                ]
            ]
        )
        try:
            await context.bot.send_message(
                ADMIN_ID,
                "🆕 <b>Новая заявка</b>\n"
                f"Пользователь: @{escape(user.username or '—')} (<code>{user.id}</code>)\n"
                f"ID Pocket Option: <code>{escape(po_id)}</code>",
                parse_mode=ParseMode.HTML,
                reply_markup=kb,
            )
        except Exception as e:
            log.warning("Не удалось отправить заявку админу: %s", e)


async def uid_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        context.user_data["await_uid"] = True
        await update.message.reply_text("🪪 Пришлите ваш ID из профиля Pocket Option (только цифры).")
        return
    await submit_uid(update, context, context.args[0])


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get("await_uid"):
        await submit_uid(update, context, update.message.text or "")


async def users_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        return
    data = load_data()
    users = data["users"]
    counts = {"approved": 0, "pending": 0, "rejected": 0}
    for u in users.values():
        counts[u.get("status", "pending")] = counts.get(u.get("status", "pending"), 0) + 1
    lines = [
        f"{'✅' if u['status'] == 'approved' else '⏳' if u['status'] == 'pending' else '❌'} "
        f"@{escape(u['name'])} · <code>{uid}</code> · PO: <code>{escape(u['po_id'])}</code>"
        for uid, u in users.items()
    ]
    await update.message.reply_text(
        f"<b>👥 Пользователи</b>\nПодтверждено: {counts['approved']} · "
        f"Ожидают: {counts['pending']} · Отклонено: {counts['rejected']}\n\n"
        + ("\n".join(lines) if lines else "Пока никого нет."),
        parse_mode=ParseMode.HTML,
    )


async def decide(context: ContextTypes.DEFAULT_TYPE, uid: int, ok: bool) -> bool:
    data = load_data()
    rec = data["users"].get(str(uid))
    if not rec:
        return False
    rec["status"] = "approved" if ok else "rejected"
    save_data(data)
    try:
        if ok:
            await context.bot.send_message(
                uid,
                "✅ <b>Регистрация подтверждена!</b>\nДоступ открыт — нажмите /start.",
                parse_mode=ParseMode.HTML,
            )
        else:
            await context.bot.send_message(
                uid,
                "❌ <b>Заявка отклонена.</b>\nПроверьте регистрацию по ссылке автора и отправьте ID "
                "снова (/start) или напишите в поддержку.",
                parse_mode=ParseMode.HTML,
            )
    except Exception as e:
        log.warning("Не удалось уведомить пользователя: %s", e)
    return True


async def approve_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID or not context.args:
        return
    try:
        uid = int(context.args[0])
    except ValueError:
        return
    found = await decide(context, uid, True)
    await update.message.reply_text(f"✅ {uid} подтверждён." if found else "Такой заявки нет.")


# ======================= КОМАНДЫ =======================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await user_tier(context.bot, update.effective_user.id) == "none":
        await update.message.reply_text(
            LOCKED, parse_mode=ParseMode.HTML, reply_markup=subscribe_keyboard()
        )
        return
    data = load_data()
    st = status_of(data, update.effective_user.id)
    if st != "approved":
        await update.message.reply_text(
            gate_text(st),
            parse_mode=ParseMode.HTML,
            reply_markup=register_keyboard() if st != "pending" else None,
        )
        return
    await update.message.reply_text(welcome_text(), parse_mode=ParseMode.HTML, reply_markup=main_menu())


# ======================= РЕЗУЛЬТАТ =======================
def build_result_text(pair: str, tf_name: str, expiry: str, res: dict) -> str:
    decimals = 3 if "JPY" in pair else 5
    p_up = res["p_up"]
    p_down = 1 - p_up
    edge = res["edge_pp"]

    if res["direction"] == "UP":
        head = "🟢 <b>▲ ВВЕРХ (CALL)</b>"
    else:
        head = "🔴 <b>▼ ВНИЗ (PUT)</b>"

    ae = abs(edge)
    edge_word = "практически ноль" if ae < 0.5 else ("ничтожный" if ae < 1.5 else "небольшой")

    parts_txt = "\n".join(
        f"  ◆ {escape(name)}\n      <code>{raw * 100:5.1f}%</code> → <code>{sh * 100:5.2f}%</code>"
        for name, raw, sh in res["parts"]
    )
    rho_state = (
        "внутри полосы шума" if abs(res["rho1"]) <= res["noise_band"] else "вне полосы (вероятно, случайность)"
    )

    if res["source"] == "live":
        source = "📡 Источник: рыночные котировки"
        price_line = f"  ┣ Цена: <code>{res['price']:.{decimals}f}</code>\n"
    else:
        source = "🌙 <b>OTC-режим</b>: рынок закрыт, расчёт по модели без реальных котировок"
        price_line = ""

    now = datetime.now(TZ).strftime("%d.%m.%Y %H:%M МСК")
    return (
        f"{TITLE}\n"
        f"<b>📊 {pair}{' OTC' if res['source'] == 'otc' else ''} · {tf_name}</b>\n"
        f"🕒 <code>{now}</code>\n"
        f"{source}\n"
        f"{LINE}\n"
        f"🎯 <b>СИГНАЛ «50/50»</b>\n"
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
        f"{price_line}"
        f"  ┣ Волатильность: {res['volatility']} (σ≈{res['sigma_pct']:.3f}%)\n"
        f"  ┗ Ориентир по экспирации: {expiry}\n"
        f"{LINE}\n"
        f"{SHORT_DISCLAIMER}"
    )


# ======================= КНОПКИ =======================
async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    user = q.from_user
    action = q.data

    # --- решения администратора ---
    if action.startswith("adm:"):
        if user.id != ADMIN_ID:
            await q.answer()
            return
        _, verdict, uid = action.split(":")
        found = await decide(context, int(uid), verdict == "ok")
        await q.answer("Готово" if found else "Заявка не найдена")
        mark = "✅ Подтверждено" if verdict == "ok" else "❌ Отклонено"
        try:
            await q.edit_message_text(
                (q.message.text_html or "") + f"\n\n<b>{mark}</b>", parse_mode=ParseMode.HTML
            )
        except BadRequest:
            pass
        return

    await q.answer()
    tier = await user_tier(context.bot, user.id)
    if tier == "none":
        await safe_edit(q, LOCKED, subscribe_keyboard())
        return

    data = load_data()
    st = status_of(data, user.id)

    # --- регистрация ---
    if action == "send_id":
        context.user_data["await_uid"] = True
        await safe_edit(
            q,
            f"{TITLE}\n{LINE}\n🪪 <b>Отправьте ваш ID</b>\n\n"
            "Скопируйте ID из профиля Pocket Option и пришлите его следующим сообщением "
            "(только цифры).",
            InlineKeyboardMarkup(
                [[InlineKeyboardButton("🚀 Открыть Pocket Option", url=REF_LINK)]]
            ),
        )
        return

    if st != "approved":
        await safe_edit(q, gate_text(st), register_keyboard() if st != "pending" else None)
        return

    if action == "menu":
        await safe_edit(q, welcome_text(), main_menu())

    elif action == "risks":
        await safe_edit(q, RISKS, InlineKeyboardMarkup([back_row()]))

    elif action == "status":
        text = (
            "<b>👤 ПРОФИЛЬ</b>\n"
            f"{LINE}\n"
            f"  ┣ ID Telegram: <code>{user.id}</code>\n"
            f"  ┣ Канал: <b>{'💎 VIP' if tier == 'vip' else '🆓 Основной'}</b>\n"
            f"  ┣ Регистрация: <b>✅ подтверждена</b>\n"
            f"  ┗ Сигналы: <b>без ограничений</b>\n"
            f"{LINE}"
        )
        await safe_edit(q, text, InlineKeyboardMarkup([back_row()]))

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
            "<b>📊 НОВЫЙ СИГНАЛ</b>\n"
            f"{LINE}\n"
            "Шаг <b>1</b> из <b>2</b> — выберите валютную пару:",
            InlineKeyboardMarkup(rows),
        )

    elif action.startswith("pair:"):
        pair = action[5:]
        if pair not in PAIRS:
            return
        row = [
            InlineKeyboardButton(f"⏱ {v[0]} · экспирация {v[1]}", callback_data=f"sig:{pair}:{code}")
            for code, v in TIMEFRAMES.items()
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

        tf_name, expiry = TIMEFRAMES[tf][0], TIMEFRAMES[tf][1]
        await safe_edit(q, f"⏳ <b>{pair} · {tf_name}</b>\n🧮 Загружаю данные и запускаю симуляцию…")

        res = await asyncio.to_thread(get_signal, pair, tf)

        text = build_result_text(pair, tf_name, expiry, res)
        kb = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("🔄 Обновить", callback_data=f"sig:{pair}:{tf}")],
                [
                    InlineKeyboardButton("🕒 Другой таймфрейм", callback_data=f"pair:{pair}"),
                    InlineKeyboardButton("💱 Другая пара", callback_data="pairs"),
                ],
                [InlineKeyboardButton("⬅️ Меню", callback_data="menu")],
            ]
        )
        await safe_edit(q, text, kb)


def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(CommandHandler("uid", uid_cmd))
    app.add_handler(CommandHandler("approve", approve_cmd))
    app.add_handler(CommandHandler("users", users_cmd))
    app.add_handler(CallbackQueryHandler(on_button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.run_polling()


if __name__ == "__main__":
    main()
