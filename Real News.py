import os
import re
import json
import time
import html
import logging
import hashlib
from datetime import datetime, timedelta, timezone
import feedparser
import requests
import schedule
from dotenv import load_dotenv
from openai import OpenAI

# =========================
# LOAD ENV
# =========================
load_dotenv()

# =========================
# CONFIG
# =========================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

RSS_URL = "https://news.google.com/rss/search?q=stock+market+india+OR+nifty+OR+sensex+OR+bank+nifty&hl=en-IN&gl=IN&ceid=IN:en"

ALTERNATE_FEEDS = [
    "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "https://www.moneycontrol.com/rss/marketreports.xml",
    "https://www.cnbctv18.com/commonfeeds/v1/cne/rss/market.xml",
]

POLL_MINUTES = int(os.getenv("POLL_MINUTES", "10"))
FRESH_HOURS = int(os.getenv("FRESH_HOURS", "12"))
MAX_NEWS_PER_RUN = int(os.getenv("MAX_NEWS_PER_RUN", "20"))
MIN_IMPACT_TO_SEND = int(os.getenv("MIN_IMPACT_TO_SEND", "50"))
OPENAI_CALLS_LIMIT_PER_DAY = int(os.getenv("OPENAI_CALLS_LIMIT_PER_DAY", "200"))

if not TELEGRAM_TOKEN or not CHAT_ID:
    raise RuntimeError("TELEGRAM_TOKEN or CHAT_ID not set in .env")

OPENAI_ENABLED = bool(OPENAI_API_KEY)

# =========================
# LOGGING + SESSION
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; IndianMarketNewsBot/3.0; +https://github.com)",
    "Accept": "application/rss+xml, text/xml",
    "Accept-Language": "en-IN,en;q=0.9"
})

client = OpenAI(api_key=OPENAI_API_KEY, timeout=30) if OPENAI_ENABLED else None

# =========================
# MEMORY (with size limit)
# =========================
seen_news = set()
openai_calls_today = 0
openai_day_marker = datetime.now().strftime("%Y-%m-%d")

# =========================
# HELPERS
# =========================
def clean_text(text):
    text = html.unescape(text or "")
    return re.sub(r"\s+", " ", text).strip()

def make_id(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def reset_daily_counter_if_needed():
    global openai_calls_today, openai_day_marker
    today = datetime.now().strftime("%Y-%m-%d")
    if today != openai_day_marker:
        openai_day_marker = today
        openai_calls_today = 0
        logging.info("🔄 OpenAI daily counter reset")

def is_fresh(entry):
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        try:
            pub_dt = datetime.fromtimestamp(time.mktime(entry.published_parsed), tz=timezone.utc)
            return datetime.now(timezone.utc) - pub_dt <= timedelta(hours=FRESH_HOURS)
        except:
            return True
    return True

def has_market_keyword(text):
    t = text.lower()
    keywords = ["nifty","sensex","bank nifty","rbi","fii","dii","repo","earnings","q4","ipo","sebi",
                "inflation","budget","fed","gdp","crude oil","tariff"]
    return any(k in t for k in keywords)

def detect_sector(text):
    t = text.lower()
    sector_map = {
        "Banking": ["bank", "rbi", "repo", "nbfc", "hdfc", "icici", "sbi"],
        "IT": ["it", "tcs", "infosys", "wipro", "tech", "hcl"],
        "Auto": ["auto", "maruti", "tata motors", "mahindra", "ev"],
        "Oil & Gas": ["oil", "crude", "ongc", "bpcl", "hpcl"],
        "Pharma": ["pharma", "drug", "sun pharma", "cipla"],
    }
    for sector, words in sector_map.items():
        if any(w in t for w in words):
            return sector
    return "General"

def get_fallback_data(title):
    return {
        "india_relevant": True,
        "sentiment": "NEUTRAL",
        "impact": 55,
        "impact_label": "MEDIUM",
        "market_view": "WAIT",
        "sector": detect_sector(title),
        "reason": "Fallback mode (OpenAI quota / disabled)",
        "summary_1line": title[:85]
    }

def ask_openai(title, summary, link):
    global openai_calls_today
    reset_daily_counter_if_needed()

    if not OPENAI_ENABLED or not client or openai_calls_today >= OPENAI_CALLS_LIMIT_PER_DAY:
        return get_fallback_data(title)

    prompt = f"""You are an expert Indian stock market analyst.
Analyze this news and return **ONLY** valid JSON (no extra text, no markdown).

Title: {title}
Summary: {summary}
Link: {link}

{{
  "india_relevant": true,
  "sentiment": "BULLISH",
  "impact": 65,
  "impact_label": "MEDIUM",
  "market_view": "BUY",
  "sector": "Banking",
  "reason": "Short reason max 20 words",
  "summary_1line": "One line summary max 15 words"
}}

Rules: sentiment = BULLISH / BEARISH / NEUTRAL
market_view = BUY / SELL / WAIT
impact = 0 to 100
"""

    try:
        openai_calls_today += 1
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").replace("json", "", 1).strip()

        data = json.loads(raw)

        return {
            "india_relevant": bool(data.get("india_relevant", True)),
            "sentiment": str(data.get("sentiment", "NEUTRAL")).upper(),
            "impact": max(0, min(100, int(data.get("impact", 50)))),
            "impact_label": str(data.get("impact_label", "MEDIUM")).upper(),
            "market_view": str(data.get("market_view", "WAIT")).upper(),
            "sector": clean_text(data.get("sector", detect_sector(title))),
            "reason": clean_text(data.get("reason", "Market moving news")),
            "summary_1line": clean_text(data.get("summary_1line", title[:80]))
        }
    except Exception as e:
        logging.exception("OpenAI error")
        return get_fallback_data(title)

def format_message(title, published, link, ai_data):
    impact_emoji = "🔴" if ai_data["impact"] >= 80 else "🟠" if ai_data["impact"] >= 60 else "🟡"

    signal = ("📈 STRONG BUY" if ai_data["sentiment"] == "BULLISH" and ai_data["impact"] >= 75 else
              "📉 STRONG SELL" if ai_data["sentiment"] == "BEARISH" and ai_data["impact"] >= 75 else
              "🔺 BUY" if ai_data["market_view"] == "BUY" else
              "🔻 SELL" if ai_data["market_view"] == "SELL" else "⚖️ WAIT")

    return f"""
<b>{signal}</b> {impact_emoji}

📰 <b>{html.escape(title)}</b>
🧠 <b>Summary:</b> {html.escape(ai_data["summary_1line"])}
🇮🇳 <b>India Relevant:</b> Yes
📊 <b>Impact:</b> {ai_data["impact"]}% ({ai_data["impact_label"]})
💹 <b>Sentiment:</b> {ai_data["sentiment"]}
🏭 <b>Sector:</b> {ai_data["sector"]}
📝 <b>Reason:</b> {html.escape(ai_data["reason"])}
🕒 {html.escape(published)}
🔗 {html.escape(link)}
""".strip()

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML", "disable_web_page_preview": False}
    try:
        r = session.post(url, data=payload, timeout=15)
        if r.status_code != 200:
            logging.error(f"Telegram failed: {r.status_code} - {r.text}")
    except Exception as e:
        logging.exception("Telegram error")

# =========================
# MAIN BOT
# =========================
def run_bot():
    global seen_news
    reset_daily_counter_if_needed()
    logging.info("🔍 Fetching Indian Market News...")

    feeds_to_check = [RSS_URL] + ALTERNATE_FEEDS[:2]
    all_entries = []

    for url in feeds_to_check:
        try:
            feed = feedparser.parse(url)
            all_entries.extend(feed.entries)
            logging.info(f"✓ Fetched {len(feed.entries)} from {url.split('//')[1].split('/')[0]}")
        except Exception as e:
            logging.error(f"✗ Failed {url}: {e}")

    # Limit + remove old duplicates
    entries = all_entries[:MAX_NEWS_PER_RUN * 3]
    sent_count = 0

    for entry in entries:
        title = clean_text(getattr(entry, "title", ""))
        if not title:
            continue

        news_id = make_id(title)
        if news_id in seen_news:
            continue
        seen_news.add(news_id)

        # Keep seen_news size in check
        if len(seen_news) > 1500:
            seen_news.clear()   # simple reset every ~1500 news

        if not is_fresh(entry):
            continue

        summary = clean_text(getattr(entry, "summary", entry.get("description", "")))
        link = clean_text(getattr(entry, "link", ""))
        published = clean_text(entry.get("published", "Recent"))

        if not has_market_keyword(f"{title} {summary}"):
            continue

        ai_data = ask_openai(title, summary, link)

        if not ai_data.get("india_relevant", True) or ai_data["impact"] < MIN_IMPACT_TO_SEND:
            continue

        msg = format_message(title, published, link, ai_data)
        send_telegram(msg)
        sent_count += 1
        logging.info(f"✅ Sent: {title[:60]}...")
        time.sleep(1.5)   # Telegram safe

    logging.info(f"🏁 Run completed | Alerts: {sent_count} | OpenAI calls today: {openai_calls_today}/{OPENAI_CALLS_LIMIT_PER_DAY}")

# =========================
# START
# =========================
if __name__ == "__main__":
    logging.info("🚀 Indian Market News Bot v3.0 Started (PythonAnywhere Optimized)")

    startup_msg = f"""🚀 <b>Market News Bot Restarted</b>
📡 Sources: Google News + ET + Moneycontrol + CNBC
⏰ Polling: Every {POLL_MINUTES} minutes
🤖 OpenAI: {"Enabled" if OPENAI_ENABLED else "Disabled (Fallback Mode)"}
🔥 Daily OpenAI Limit: {OPENAI_CALLS_LIMIT_PER_DAY}"""

    send_telegram(startup_msg)

    run_bot()
if __name__ == "__main__":
    run_bot()
