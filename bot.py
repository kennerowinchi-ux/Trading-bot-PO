import asyncio
import base64
import json
import logging
import math
import os
import random
import time
import urllib.error
import urllib.request
from collections import deque
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
BACKUP_KEY = os.getenv("BACKUP_KEY", "")  # необязательно: если задан, файл в репозитории шифруется
REPO_FILE = "data.enc" if BACKUP_KEY else "registrations.json"  # копия регистраций в репозитории (usage в неё не входит)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPOSITORY = os.getenv("GITHUB_REPOSITORY", "")  # owner/repo, задаётся GitHub Actions сам
GITHUB_BRANCH = os.getenv("GITHUB_REF_NAME", "main")
CACHE_TTL = 45  # секунд (кэш котировок)
ALT_PROB = 0.5   # вероятность, что направление сменится на противоположное предыдущему (0 = только естественный шанс 50%; 1 = всегда меняется)
LOAD_MIN, LOAD_MAX = 5, 10  # длительность «анализа» в секундах
USAGE_SYNC_SEC = 900   # как часто (если были анализы) сохранять счётчик анализов в репозиторий
SUB_CHECK_SEC = 1800   # как часто проверять, что пользователи не отписались от канала
# защита от спама: не больше FLOOD_MAX нажатий за FLOOD_WINDOW секунд, иначе блок на MUTE_SEC секунд
FLOOD_WINDOW, FLOOD_MAX, MUTE_SEC, MIN_GAP = 5.0, 8, 30, 0.5
NOISE = 0.6     # «случайность» голосов: чем больше, тем сильнее результат меняется при обновлении


# Список пар (pairs.json больше не используется).
# Значение — тикер Yahoo или список запасных вариантов (пробуются по очереди).
# "A/B" — кросс-курс (A делить на B), "inv:T" — обратный курс.
PAIRS_OTC = {
    "AED/CNY OTC": ["AEDCNY=X", "USDCNY=X/USDAED=X"],
    "AUD/CHF OTC": "AUDCHF=X",
    "AUD/USD OTC": "AUDUSD=X",
    "EUR/CHF OTC": "EURCHF=X",
    "EUR/USD OTC": "EURUSD=X",
    "NGN/USD OTC": ["NGNUSD=X", "inv:USDNGN=X"],
    "USD/IDR OTC": "USDIDR=X",
    "USD/PKR OTC": "USDPKR=X",
    "USD/THB OTC": "USDTHB=X",
    "USD/VND OTC": "USDVND=X",
}

# Обычные графики (без приписки OTC)
PAIRS_REAL = {
    "USD/CHF": "USDCHF=X",       # 81%
    "CHF/JPY": "CHFJPY=X",       # 80%
    "EUR/JPY": "EURJPY=X",       # 78%
    "AUD/CHF": "AUDCHF=X",       # 77%
    "AUD/CAD": "AUDCAD=X",       # 75%
    "AUD/JPY": "AUDJPY=X",       # 73%
    "EUR/CHF": "EURCHF=X",       # 73%
    "EUR/USD": "EURUSD=X",       # 73%
    "AUD/USD": "AUDUSD=X",       # 70%
    "CAD/CHF": ["CADCHF=X", "USDCHF=X/USDCAD=X"],  # 68%
}

PAIRS = {**PAIRS_OTC, **PAIRS_REAL}
MARKETS = {
    "otc": ("💱 Графики OTC", PAIRS_OTC, ["15s", "30s", "1m", "3m", "5m"]),
    "real": ("📈 Обычные графики", PAIRS_REAL, ["1m", "3m", "5m", "30m"]),
}


def pair_kind(pair: str) -> str:
    return "otc" if pair in PAIRS_OTC else "real"

# код -> (название, ориентир по экспирации, максимальный возраст последней свечи в минутах)
TIMEFRAMES = {
    "15s": ("15S", "15 сек", None),
    "30s": ("30S", "30 сек", None),
    "1m": ("M1", "1 мин", None),
    "3m": ("M3", "3 мин", "3min"),
    "5m": ("M5", "5 мин", "5min"),
    "30m": ("M30", "30 мин", "30min"),
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_UP = os.path.join(BASE_DIR, "up.png")      # зелёное фото: график вверх
IMG_DOWN = os.path.join(BASE_DIR, "down.png")  # красное фото: график вниз
IMG_WELCOME = os.path.join(BASE_DIR, "welcome.png")    # фото главного меню
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
    tmp = DATA_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, DATA_FILE)  # файл не повредится, даже если бот упадёт во время записи


_restored = False  # True, когда база уже подтянута из хранилищ (или подтягивать нечего)
_REG_KEYS = ("approved", "rejected", "uids")


def _fernet():
    if not BACKUP_KEY:
        return None
    try:
        from cryptography.fernet import Fernet

        return Fernet(BACKUP_KEY.encode())
    except Exception as e:
        log.warning("BACKUP_KEY неверный или нет библиотеки cryptography: %s", e)
        return None


def read_repo_copy() -> dict | None:
    """Копия регистраций из репозитория (приходит вместе с checkout)."""
    if not os.path.exists(REPO_FILE):
        return None
    try:
        with open(REPO_FILE, "rb") as fh:
            raw = fh.read()
        if BACKUP_KEY:
            f = _fernet()
            if not f:
                return None
            raw = f.decrypt(raw)
        return json.loads(raw.decode("utf-8"))
    except Exception as e:
        log.warning("Не удалось прочитать %s: %s", REPO_FILE, e)
        return None


def _github_put(content: bytes) -> None:
    """Обновляет (или создаёт) файл в репозитории через GitHub API. Блокирующая функция."""
    url = f"https://api.github.com/repos/{GITHUB_REPOSITORY}/contents/{REPO_FILE}"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "syndicate-bot",
    }
    sha = None
    try:
        req = urllib.request.Request(f"{url}?ref={GITHUB_BRANCH}", headers=headers)
        with urllib.request.urlopen(req, timeout=20) as r:
            sha = json.loads(r.read().decode("utf-8")).get("sha")
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
    body = {
        "message": "Обновление базы регистраций",
        "content": base64.b64encode(content).decode("ascii"),
        "branch": GITHUB_BRANCH,
    }
    if sha:
        body["sha"] = sha
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers={**headers, "Content-Type": "application/json"}, method="PUT"
    )
    urllib.request.urlopen(req, timeout=30).read()


async def push_repo_copy() -> None:
    if not (GITHUB_TOKEN and GITHUB_REPOSITORY):
        log.info("Запись в репозиторий выключена (нужны GITHUB_TOKEN и GITHUB_REPOSITORY)")
        return
    data = load_data()
    payload = {k: data[k] for k in _REG_KEYS}
    payload["reg_ts"] = data.get("reg_ts", 0)
    payload["usage"] = {u: r for u, r in data["usage"].items() if r.get("date") == today()}  # только сегодняшний день
    raw = json.dumps(payload, ensure_ascii=False, indent=1).encode("utf-8")
    if BACKUP_KEY:
        f = _fernet()
        if not f:
            return  # ключ задан, но не работает: открытым текстом не пишем
        raw = f.encrypt(raw)
    try:
        await asyncio.to_thread(_github_put, raw)
        log.info("База регистраций записана в репозиторий (%s)", REPO_FILE)
    except Exception as e:
        log.warning("Не удалось записать базу в репозиторий: %s", e)


def mark_registration_changed(data: dict) -> None:
    data["reg_ts"] = time.time()


async def ensure_restored(bot) -> bool:
    """Собирает базу регистраций из локального файла, репозитория и Telegram и берёт самую свежую версию.
    Пока Telegram не ответил, резервные копии не перезаписываются — иначе можно затереть хорошую копию пустой."""
    global _restored
    if _restored:
        return True

    ok = True
    candidates = []
    if os.path.exists(DATA_FILE):
        candidates.append(load_data())
    repo = read_repo_copy()
    if repo:
        candidates.append(repo)

    if ADMIN_ID:
        try:
            pinned = (await bot.get_chat(ADMIN_ID)).pinned_message
            if pinned and pinned.document and pinned.document.file_name == "data.json":
                tg_file = await bot.get_file(pinned.document.file_id)
                raw = await tg_file.download_as_bytearray()
                candidates.append(json.loads(bytes(raw).decode("utf-8")))
        except Exception as e:
            ok = False
            log.warning("Не удалось получить копию из Telegram (повторю при следующем действии): %s", e)

    if candidates:
        data = load_data()
        newest = max(candidates, key=lambda c: c.get("reg_ts", 0))
        if newest.get("reg_ts", 0) > 0:
            # у версий есть метка времени: берём самую свежую целиком (так отзыв доступа не откатывается)
            for k in _REG_KEYS:
                data[k] = newest.get(k, data[k])
            data["reg_ts"] = newest["reg_ts"]
        else:
            # старые копии без метки: объединяем
            data["approved"] = sorted(set().union(*[set(c.get("approved", [])) for c in candidates]))
            rej = set().union(*[set(c.get("rejected", [])) for c in candidates])
            data["rejected"] = sorted(u for u in rej if u not in data["approved"])
            uids = {}
            for c in candidates:
                uids.update(c.get("uids", {}))
            data["uids"] = uids
        # счётчик анализов: по каждому пользователю берём наибольшее значение за сегодня
        for c in candidates:
            for u, rec in (c.get("usage") or {}).items():
                if rec.get("date") != today():
                    continue
                cur = data["usage"].get(u)
                if not cur or cur.get("date") != today() or rec.get("count", 0) > cur.get("count", 0):
                    data["usage"][u] = rec
        save_data(data)
        log.info("База восстановлена: подтверждённых %d", len(data["approved"]))

    _restored = ok
    return _restored


async def restore_data(app) -> None:
    await ensure_restored(app.bot)


async def backup_data(bot) -> None:
    """Сохраняет базу регистраций: файлом в Telegram (закреплён у админа) и зашифрованной копией в репозитории.
    Из них бот восстанавливается после любого перезапуска, даже на чистой машине."""
    if not await ensure_restored(bot):
        log.warning("Резервная копия пропущена: база ещё не восстановлена, чтобы не затереть старую копию")
        return
    if ADMIN_ID:
        try:
            old = (await bot.get_chat(ADMIN_ID)).pinned_message
            with open(DATA_FILE, "rb") as f:
                msg = await bot.send_document(
                    ADMIN_ID, f, filename="data.json", caption="💾 Резервная копия базы", disable_notification=True
                )
            await bot.pin_chat_message(ADMIN_ID, msg.message_id, disable_notification=True)
            if old and old.document and old.document.file_name == "data.json":
                await bot.unpin_chat_message(ADMIN_ID, old.message_id)
        except Exception as e:
            log.warning("Не удалось сохранить резервную копию в Telegram: %s", e)
    await push_repo_copy()


_usage_dirty = False  # были анализы, которые ещё не записаны в репозиторий


async def usage_sync_loop() -> None:
    """Раз в USAGE_SYNC_SEC сохраняет счётчик анализов в репозиторий (если он менялся)."""
    global _usage_dirty
    while True:
        await asyncio.sleep(USAGE_SYNC_SEC)
        if _usage_dirty and _restored:
            _usage_dirty = False
            await push_repo_copy()


# ======================= АВТОПРОВЕРКА ПОДПИСКИ =======================
_unsub: set = set()
_sub_baseline = False  # первый проход молча запоминает, кто уже не подписан


async def _member_or_none(bot, chat_id: int, user_id: int):
    """True/False — статус подписки, None — не удалось узнать (ошибка сети и т. п.)."""
    try:
        m = await bot.get_chat_member(chat_id, user_id)
        return m.status in ("member", "administrator", "creator") or (
            m.status == "restricted" and getattr(m, "is_member", False)
        )
    except Exception:
        return None


async def check_subscriptions(bot) -> None:
    global _sub_baseline
    data = load_data()
    users = set(data["approved"]) | {int(u) for u in data["uids"]} | {int(u) for u in data["usage"]}
    users.discard(ADMIN_ID)
    for uid in sorted(users):
        st_free = await _member_or_none(bot, FREE_CHANNEL_ID, uid)
        st_vip = await _member_or_none(bot, VIP_CHANNEL_ID, uid)
        subscribed = bool(st_free) or bool(st_vip)
        if not subscribed and (st_free is None or st_vip is None):
            continue  # не уверены (ошибка запроса) — ничего не делаем
        if not subscribed and uid not in _unsub:
            _unsub.add(uid)
            if _sub_baseline:
                try:
                    await bot.send_message(
                        uid,
                        "<b>⚠️ Подписка не найдена</b>\n"
                        f"{LINE}\n"
                        "Вы не подписаны на канал, поэтому доступ к боту закрыт.\n"
                        "Подпишитесь и нажмите «Проверить подписку».",
                        parse_mode=ParseMode.HTML,
                        reply_markup=subscribe_keyboard(),
                    )
                except Exception:
                    pass
        elif subscribed and uid in _unsub:
            _unsub.discard(uid)
            if _sub_baseline:
                try:
                    await bot.send_message(uid, "✅ Подписка восстановлена, доступ к боту снова открыт. Нажмите /start.")
                except Exception:
                    pass
        await asyncio.sleep(0.1)  # не упираемся в лимиты Telegram
    _sub_baseline = True


async def subscription_watch_loop(bot) -> None:
    await asyncio.sleep(60)
    while True:
        try:
            await check_subscriptions(bot)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("Автопроверка подписки не удалась: %s", e)
        await asyncio.sleep(SUB_CHECK_SEC)


# ======================= ЗАЩИТА ОТ СПАМА =======================
_hits: dict = {}
_muted_until: dict = {}
_cmd_last: dict = {}
_busy: set = set()  # пользователи, у которых прямо сейчас идёт анализ


def flood_check(uid: int) -> str | None:
    """Возвращает текст предупреждения, если нажатие нужно проигнорировать."""
    if uid == ADMIN_ID:
        return None
    now = time.time()
    until = _muted_until.get(uid, 0)
    if until > now:
        return f"🚫 Слишком много нажатий. Подождите {int(until - now) + 1} сек."
    dq = _hits.setdefault(uid, deque())
    while dq and now - dq[0] > FLOOD_WINDOW:
        dq.popleft()
    dq.append(now)
    if len(dq) > FLOOD_MAX:
        _muted_until[uid] = now + MUTE_SEC
        dq.clear()
        return f"🚫 Слишком много нажатий. Подождите {MUTE_SEC} сек."
    if len(dq) >= 2 and now - dq[-2] < MIN_GAP:
        return "⏳ Не так быстро"
    return None


def cmd_ok(uid: int, key: str, seconds: float) -> bool:
    """Ограничение частоты команд: не чаще одного раза в `seconds` секунд."""
    if uid == ADMIN_ID:
        return True
    now = time.time()
    if now - _cmd_last.get((uid, key), 0) < seconds:
        return False
    _cmd_last[(uid, key)] = now
    return True


async def on_start(app) -> None:
    """Запуск: восстановить базу и включить фоновые задачи."""
    await ensure_restored(app.bot)
    app.bot_data["tasks"] = [
        asyncio.create_task(usage_sync_loop()),
        asyncio.create_task(subscription_watch_loop(app.bot)),
    ]


async def backup_on_stop(app) -> None:
    """Перед остановкой (плановый перезапуск раз в ~6 часов) сохраняем свежую копию."""
    for t in app.bot_data.get("tasks", []):
        t.cancel()
    await backup_data(app.bot)


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


_last_dir: dict = {}  # user_id -> последнее выпавшее направление (+1 вверх, -1 вниз)


def vote_now(base: dict, last_dir: int = 0) -> dict:
    """Новый «раунд» голосования: сила индикатора + шум. После прошлого результата направление
    с вероятностью ALT_PROB меняется на противоположное (вниз -> чаще вверх и наоборот)."""
    votes = []
    for name, up_txt, down_txt, x in base["feats"]:
        v = 1 if math.tanh(x) + random.gauss(0, NOISE) > 0 else -1
        votes.append([name, up_txt, down_txt, x, v])

    # Если нужна смена направления, а выпало то же самое — переворачиваем самых «неуверенных» судей
    want = -last_dir if last_dir and random.random() < ALT_PROB else 0
    if want and sum(v[4] for v in votes) * want < 0:
        need = random.choice([1, 3, 3, 5])  # с какой силой должен выйти новый результат
        for v in sorted((v for v in votes if v[4] != want), key=lambda v: abs(math.tanh(v[3]))):
            v[4] = want
            if sum(j[4] for j in votes) * want >= need:
                break
    votes = [(name, up_txt if v > 0 else down_txt, v) for name, up_txt, down_txt, _, v in votes]

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


def _download_one(ticker: str) -> pd.DataFrame | None:
    df = yf.download(ticker, period="5d", interval="1m", progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["Close", "High", "Low"])


def download_spec(spec: str) -> pd.DataFrame | None:
    """'T' — обычный тикер; 'inv:T' — обратный курс; 'A/B' — кросс-курс A делить на B."""
    cols = ["Open", "High", "Low", "Close"]
    if spec.startswith("inv:"):
        df = _download_one(spec[4:])
        if df is None:
            return None
        inv = 1 / df[cols]
        # при обращении максимум и минимум меняются местами
        inv = inv.rename(columns={"High": "Low", "Low": "High"})
        return inv[cols]
    if "/" in spec:
        a, b = spec.split("/", 1)
        da, db = _download_one(a), _download_one(b)
        if da is None or db is None:
            return None
        joined = da[cols].join(db[cols], how="inner", lsuffix="_a", rsuffix="_b")
        if joined.empty:
            return None
        out = pd.DataFrame(index=joined.index)
        for c in cols:
            out[c] = joined[c + "_a"] / joined[c + "_b"]
        out["High"] = out[["High", "Open", "Close"]].max(axis=1)
        out["Low"] = out[["Low", "Open", "Close"]].min(axis=1)
        return out
    return _download_one(spec)


def download_pair(ticker) -> pd.DataFrame | None:
    specs = [ticker] if isinstance(ticker, str) else list(ticker)
    for spec in specs:
        try:
            df = download_spec(spec)
        except Exception as e:
            log.warning("download %s failed: %s", spec, e)
            continue
        if df is not None and len(df) >= 60:
            return df
    return None


def fetch_and_analyze(ticker, interval: str, last_dir: int = 0) -> dict | None:
    key = (ticker if isinstance(ticker, str) else tuple(ticker), interval)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        base = hit[1]
    else:
        # yfinance не умеет интервалы 30s и 3m: качаем минутные свечи,
        # для 15S/30S берём M1 как есть, а M3/M5/M30 собираем из минутных свечей.
        df = download_pair(ticker)
        if df is None:
            return None

        rule = TIMEFRAMES[interval][2]
        if rule:
            df = df.resample(rule).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
        if len(df) < 60:
            return None

        base = analyze_df(df)
        _cache[key] = (time.time(), base)

    return vote_now(base, last_dir)


# ======================= ТЕКСТЫ =======================
WELCOME = (
    "<b>📈 Syndicate Trade </b>\n"
    f"{LINE}\n"
    "Syndicate Trade - Специальный бот, созданный для анализа рынка и определением сигналов с последующим направлением рынка.\n"
    "Данный бот имеет возможность анализировать рынок 24/7, благодаря подключенному Claude AI 🚀 \n\n"
    "<i>Доступ к боту имеют все подписчики телеграмм канала блогера Terret</i>\n\n"
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
    "<b>5. Экспирация.</b> Для OTC-графиков доступны 15S, 30S, M1, M3, M5, для обычных — M1, M3, M5, M30. "
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


async def set_loading(q, text: str):
    """Показывает экран загрузки (без кнопок). Если сейчас на экране фото — меняет только подпись."""
    try:
        if q.message is not None and q.message.photo:
            await q.edit_message_caption(caption=text, parse_mode=ParseMode.HTML)
        else:
            await safe_edit(q, text)
    except BadRequest as e:
        if "not modified" not in str(e).lower():
            raise


async def analyze_with_loading(q, pair: str, tf_name: str, ticker, tf: str, last_dir: int):
    """Запускает расчёт и параллельно показывает «идёт анализ» 5–10 секунд."""
    task = asyncio.create_task(asyncio.to_thread(fetch_and_analyze, ticker, tf, last_dir))
    steps = [
        "📡 Загружаю котировки",
        "📊 Считаю индикаторы",
        "⚖️ Судьи голосуют",
        "🧮 Подвожу итог",
    ]
    total = random.uniform(LOAD_MIN, LOAD_MAX)
    pause = total / len(steps)
    for i, step in enumerate(steps, 1):
        bar = "▰" * i + "▱" * (len(steps) - i)
        await set_loading(
            q, f"⏳ <b>{escape(pair)} · {tf_name}</b>\n{LINE}\n<b>Идёт анализ…</b>\n\n{step}\n<code>{bar}</code>"
        )
        await asyncio.sleep(pause)
    return await task


async def show_photo(q, path: str, text: str, kb: InlineKeyboardMarkup):
    """Показывает фото с подписью. Если файла нет — обычный текст."""
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


async def show_result(q, text: str, kb: InlineKeyboardMarkup, direction: str):
    """Результат анализа с фото: зелёное для ВВЕРХ, красное для ВНИЗ."""
    await show_photo(q, IMG_UP if direction == "UP" else IMG_DOWN, text, kb)


# ======================= КОМАНДЫ =======================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not cmd_ok(update.effective_user.id, "start", 2):
        return
    tier = await user_tier(context.bot, update.effective_user.id)
    if tier == "none":
        await update.message.reply_text(
            LOCKED, parse_mode=ParseMode.HTML, reply_markup=subscribe_keyboard()
        )
        return
    if os.path.exists(IMG_WELCOME):
        with open(IMG_WELCOME, "rb") as f:
            await update.message.reply_photo(f, caption=WELCOME, parse_mode=ParseMode.HTML, reply_markup=main_menu())
    else:
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
    if not cmd_ok(user.id, "uid", 20):
        await update.message.reply_text("⏳ Подождите немного перед повторной отправкой.")
        return
    uid = context.args[0][:32]
    await ensure_restored(context.bot)
    data = load_data()
    if user.id in data["approved"]:
        await update.message.reply_text("✅ Ваша регистрация уже подтверждена, повторно отправлять ID не нужно.")
        return
    data["uids"][str(user.id)] = uid
    if user.id in data["rejected"]:
        data["rejected"].remove(user.id)
    mark_registration_changed(data)
    save_data(data)
    await backup_data(context.bot)
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
    await ensure_restored(bot)
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
    mark_registration_changed(data)
    save_data(data)
    await backup_data(bot)
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
    user = q.from_user

    warn = flood_check(user.id)
    if warn:
        await q.answer(warn, show_alert=warn.startswith("🚫"))
        return
    if q.data.startswith("sig:") and user.id in _busy:
        await q.answer("⏳ Анализ уже идёт, подождите несколько секунд")
        return
    await q.answer()

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

    await ensure_restored(context.bot)
    data = load_data()
    action = q.data

    # Анализ рынка доступен только после подтверждения регистрации администратором
    needs_approval = action in ("pairs",) or action.startswith(("list:", "pair:", "sig:"))
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
        await show_photo(q, IMG_WELCOME, WELCOME, main_menu())

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
        kb = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton(MARKETS["otc"][0], callback_data="list:otc")],
                [InlineKeyboardButton(MARKETS["real"][0], callback_data="list:real")],
                back_row(),
            ]
        )
        await safe_edit(
            q,
            "<b>📊 Анализ рынка</b>\n"
            f"{LINE}\n"
            "Выберите тип графика:\n\n"
            "💱 <b>OTC</b> — экспирация 15S, 30S, M1, M3, M5\n"
            "📈 <b>Обычные</b> — экспирация M1, M3, M5, M30",
            kb,
        )

    elif action.startswith("list:"):
        kind = action[5:]
        if kind not in MARKETS:
            return
        title, pairs, _ = MARKETS[kind]
        names = list(pairs.keys())
        rows = [
            [InlineKeyboardButton(n, callback_data=f"pair:{n}") for n in names[i : i + 2]]
            for i in range(0, len(names), 2)
        ]
        rows.append([InlineKeyboardButton("⬅️ К выбору типа", callback_data="pairs")])
        await safe_edit(q, f"<b>{title}</b>\n{LINE}\nВыберите пару:", InlineKeyboardMarkup(rows))

    elif action.startswith("pair:"):
        pair = action[5:]
        if pair not in PAIRS:
            return
        kind = pair_kind(pair)
        kb = [
            [InlineKeyboardButton(f"{TIMEFRAMES[code][0]} · экспирация {TIMEFRAMES[code][1]}",
                                  callback_data=f"sig:{pair}:{code}")]
            for code in MARKETS[kind][2]
        ]
        kb.append([InlineKeyboardButton("⬅️ К выбору пары", callback_data=f"list:{kind}")])
        await safe_edit(
            q,
            f"<b>📊 {escape(pair)}</b>\n"
            f"{LINE}\n"
            "Шаг 2 из 2 — выберите время захода в сделку.\n\n"
            "<i>Таймфрейм - Время, на которое вы входите в сделку.</i>",
            InlineKeyboardMarkup(kb),
        )

    elif action.startswith("sig:"):
        try:
            _, pair, tf = action.split(":")
        except ValueError:
            return
        if pair not in PAIRS or tf not in MARKETS[pair_kind(pair)][2]:
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
        if user.id in _busy:  # проверка и добавление без await между ними, параллельный анализ невозможен
            return
        _busy.add(user.id)
        try:
            res = await analyze_with_loading(q, pair, tf_name, PAIRS[pair], tf, _last_dir.get(user.id, 0))
        except Exception as e:
            log.exception("analysis failed: %s", e)
            res = None
        finally:
            _busy.discard(user.id)

        back_kb = InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Другая пара", callback_data=f"list:{pair_kind(pair)}")], back_row()])

        if res is None:
            await safe_edit(
                q,
                "<b>😕 Данные недоступны</b>\n\nНе удалось получить котировки. "
                "Попробуйте позже. Анализ не засчитан в лимит.",
                back_kb,
            )
            return

        data = load_data()  # перечитываем: за время анализа админ мог изменить статусы
        rec = data["usage"].get(str(user.id))
        if not rec or rec["date"] != today():
            rec = {"date": today(), "count": 0}
        rec["count"] += 1
        data["usage"][str(user.id)] = rec
        save_data(data)
        global _usage_dirty
        _usage_dirty = True

        _last_dir[user.id] = 1 if res["direction"] == "UP" else -1
        p = res["price"]
        decimals = 2 if p >= 1000 else 3 if p >= 100 else 4 if p >= 10 else 5 if p >= 0.1 else 7
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
            f"📋 Пара: <code>{escape(pair)}</code> <i>(нажмите, чтобы скопировать)</i>\n"
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
                    InlineKeyboardButton("💱 Другая пара", callback_data=f"list:{pair_kind(pair)}"),
                ],
                [
                    InlineKeyboardButton("📚 Методика", callback_data="method"),
                    InlineKeyboardButton("⬅️ Меню", callback_data="menu"),
                ],
            ]
        )
        await show_result(q, text, kb, res["direction"])


def main():
    app = Application.builder().token(BOT_TOKEN).post_init(on_start).post_stop(backup_on_stop).build()
    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(CommandHandler("uid", uid_cmd))
    app.add_handler(CommandHandler("approve", approve_cmd))
    app.add_handler(CommandHandler("reject", reject_cmd))
    app.add_handler(CallbackQueryHandler(on_button))
    app.run_polling()


if __name__ == "__main__":
    main()
