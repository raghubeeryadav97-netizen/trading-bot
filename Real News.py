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
from transformers import pipeline
from openai import OpenAI, RateLimitError

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

RSS_URL = os.getenv("RSS_URL", "https://www.livemint.com/rss/markets")
POLL_MINUTES = int(os.getenv("POLL_MINUTES", "5"))
FRESH_HOURS = int(os.getenv("FRESH_HOURS", "12"))
MAX_NEWS_PER_RUN = int(os.getenv("MAX_NEWS_PER_RUN", "25"))
MIN_IMPACT_TO_SEND = int(os.getenv("MIN_IMPACT_TO_SEND", "45"))

# Extra quota safety
OPENAI_CALLS_LIMIT_PER_DAY = int(os.getenv("OPENAI_CALLS_LIMIT_PER_DAY", "300"))
OPENAI_TIMEOUT_SECONDS = int(os.getenv("OPENAI_TIMEOUT_SECONDS", "30"))

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN not set in .env")
if not CHAT_ID:
    raise RuntimeError("CHAT_ID not set in .env")

# OpenAI optional
OPENAI_ENABLED = bool(OPENAI_API_KEY)

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# =========================
# SESSION / CLIENTS
# =========================
session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 HybridNewsBot/3.0"})

client = OpenAI(api_key=OPENAI_API_KEY, timeout=OPENAI_TIMEOUT_SECONDS) if OPENAI_ENABLED else None

# FinBERT is local model inference after download
finbert = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert"
)

# =========================
# MEMORY / RUNTIME STATE
# =========================
seen_news = set()
openai_calls_today = 0
openai_day_marker = datetime.now().strftime("%Y-%m-%d")

# =========================
# KEYWORDS
# =========================
market_keywords = [
    "rbi", "inflation", "interest rate", "repo rate", "fii", "dii", "crude oil", "war",
    "budget", "nifty", "sensex", "bank nifty", "rupee", "earnings", "ipo", "policy",
    "fed", "gdp", "recession", "stock market", "banking", "q4", "results", "dividend",
    "guidance", "profit", "loss", "margin", "oil", "tariff", "yield", "bond",
    "stocks", "shares", "markets", "equity", "mutual fund", "sebi", "brokerage"
]

india_keywords = [
    "india", "indian", "nifty", "sensex", "rupee", "rbi", "fii", "dii", "nse", "bse",
    "bank nifty", "repo", "sebi", "mospi", "lok sabha", "gst"
]

sector_map = {
    "Banking": ["bank", "banking", "rbi", "repo", "loan", "nbfc", "psu bank", "private bank"],
    "IT": ["it", "software", "tech", "digital", "tcs", "infosys", "wipro", "hcl", "ltim", "coforge"],
    "Auto": ["auto", "vehicle", "car", "ev", "tractor", "maruti", "mahindra", "tata motors", "ashok leyland"],
    "Oil & Gas": ["oil", "gas", "crude", "ongc", "ioc", "bpcl", "hpcl", "petrol", "diesel"],
    "Pharma": ["pharma", "drug", "usfda", "hospital", "medicine", "healthcare", "biotech"],
    "Metals": ["metal", "steel", "copper", "aluminium", "zinc", "mining", "ore"],
    "FMCG": ["fmcg", "consumer", "retail", "demand", "rural demand"],
    "Capital Markets": ["ipo", "listing", "nse", "bse", "broker", "brokerage", "sebi", "mutual fund"],
    "Macro": ["inflation", "gdp", "policy", "budget", "yield", "war", "fed", "tariff", "recession"]
}

# =========================
# HELPERS
# =========================
def clean_text(text):
    text = html.unescape(text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

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
            pub_ts = time.mktime(entry.published_parsed)
            pub_dt = datetime.fromtimestamp(pub_ts, tz=timezone.utc)
            age = datetime.now(timezone.utc) - pub_dt
            return age <= timedelta(hours=FRESH_HOURS)
        except Exception:
            return True
    return True

def has_market_keyword(text):
    t = text.lower()
    return any(k in t for k in market_keywords)

def is_india_relevant_rule(text):
    t = text.lower()
    return any(k in t for k in india_keywords) or "q4" in t or "results" in t or "earnings" in t

def detect_sector_rule(text):
    t = text.lower()
    hits = []
    for sector, words in sector_map.items():
        if any(w in t for w in words):
            hits.append(sector)
    return ", ".join(hits[:2]) if hits else "General"

def finbert_sentiment(text):
    try:
        result = finbert(text[:512])[0]
        raw_label = str(result["label"]).lower()
        score = float(result["score"])
        label = (
            "BULLISH" if "positive" in raw_label
            else "BEARISH" if "negative" in raw_label
            else "NEUTRAL"
        )
        return {"label": label, "confidence": round(score, 4)}
    except Exception as e:
        logging.exception("FinBERT error: %s", e)
        return {"label": "NEUTRAL", "confidence": 0.0}

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": False
    }
    try:
        r = session.post(url, data=payload, timeout=20)
        if r.status_code != 200:
            logging.error("Telegram send failed: %s | %s", r.status_code, r.text)
    except Exception as e:
        logging.exception("Telegram error: %s", e)

def get_fallback_data(title, summary):
    return {
        "india_relevant": is_india_relevant_rule(title + " " + summary),
        "sentiment": "NEUTRAL",
        "impact": 55,
        "impact_label": "MEDIUM",
        "market_view": "WAIT",
        "sector": detect_sector_rule(title),
        "reason": "Fallback mode (OpenAI disabled/quota exhausted/error)",
        "summary_1line": title[:90]
    }

def ask_openai(title, summary, link):
    global OPENAI_ENABLED, openai_calls_today

    reset_daily_counter_if_needed()

    if not OPENAI_ENABLED or not client:
        return get_fallback_data(title, summary)

    if openai_calls_today >= OPENAI_CALLS_LIMIT_PER_DAY:
        logging.warning("🚨 Daily OpenAI call limit reached. Using fallback mode.")
        return get_fallback_data(title, summary)

    prompt = f"""
You are an Indian market news analyst.
Analyze the following news for Indian equities and output ONLY valid JSON.

Title: {title}
Summary: {summary}
Link: {link}

Return exactly:
{{
  "india_relevant": true,
  "sentiment": "BULLISH",
  "impact": 78,
  "impact_label": "HIGH",
  "market_view": "BUY",
  "sector": "Banking",
  "reason": "Short reason under 25 words",
  "summary_1line": "One line under 18 words"
}}

Rules:
- Consider Indian market reaction, not global generic tone.
- sentiment must be BULLISH, BEARISH, or NEUTRAL.
- market_view must be BUY, SELL, or WAIT.
- impact must be integer from 0 to 100.
- impact_label must be LOW, MEDIUM, or HIGH.
- If not relevant to Indian markets, set india_relevant false.
- Keep response compact.
- Return JSON only.
"""

    try:
        openai_calls_today += 1
        logging.info("🧠 OpenAI call #%s today | %s", openai_calls_today, title[:80])

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise financial news classifier for Indian markets. Return JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=300
        )

        raw = response.choices[0].message.content.strip()

        if raw.startswith("```"):
            raw = raw.strip("`").replace("json", "", 1).strip()

        data = json.loads(raw)

        impact = max(0, min(100, int(data.get("impact", 0))))
        impact_label = str(data.get("impact_label", "LOW")).upper()
        if impact_label not in ["LOW", "MEDIUM", "HIGH"]:
            impact_label = "LOW"

        sentiment = str(data.get("sentiment", "NEUTRAL")).upper()
        if sentiment not in ["BULLISH", "BEARISH", "NEUTRAL"]:
            sentiment = "NEUTRAL"

        market_view = str(data.get("market_view", "WAIT")).upper()
        if market_view not in ["BUY", "SELL", "WAIT"]:
            market_view = "WAIT"

        return {
            "india_relevant": bool(data.get("india_relevant", True)),
            "sentiment": sentiment,
            "impact": impact,
            "impact_label": impact_label,
            "market_view": market_view,
            "sector": clean_text(data.get("sector", detect_sector_rule(title))),
            "reason": clean_text(data.get("reason", "No reason")),
            "summary_1line": clean_text(data.get("summary_1line", title[:90]))
        }

    except RateLimitError as e:
        err_text = str(e).lower()
        if "insufficient_quota" in err_text:
            OPENAI_ENABLED = False
            logging.error("🚨 OpenAI quota exhausted. Switching to fallback mode.")
            send_telegram(
                "⚠️ <b>OpenAI quota exhausted</b>\n"
                "Bot ab <b>rule-based fallback mode</b> mein chal raha hai.\n"
                "AI analysis temporarily band.\n"
                "Billing/credits check karo."
            )
            return get_fallback_data(title, summary)

        logging.exception("OpenAI RateLimitError: %s", e)
        return get_fallback_data(title, summary)

    except Exception as e:
        logging.exception("OpenAI error: %s", e)
        return get_fallback_data(title, summary)

def merge_sentiment(ai_sentiment, fin_label, impact):
    ai_sentiment = ai_sentiment.upper()
    fin_label = fin_label.upper()

    if ai_sentiment == fin_label:
        return ai_sentiment
    if impact >= 70:
        return ai_sentiment
    if fin_label in ["BULLISH", "BEARISH"]:
        return fin_label
    return ai_sentiment

def final_signal(final_sentiment, market_view, impact):
    if final_sentiment == "BULLISH" and impact >= 70:
        return "📈 STRONG BUY"
    if final_sentiment == "BEARISH" and impact >= 70:
        return "📉 STRONG SELL"
    if market_view == "BUY":
        return "🔺 BUY BIAS"
    if market_view == "SELL":
        return "🔻 SELL BIAS"
    return "⚖️ WAIT / SIDEWAYS"

def format_message(title, published, link, ai_data, fin_data):
    final_sent = merge_sentiment(ai_data["sentiment"], fin_data["label"], ai_data["impact"])
    signal = final_signal(final_sent, ai_data["market_view"], ai_data["impact"])

    return f"""
<b>{signal}</b>
📰 <b>Headline:</b> {html.escape(title)}
🧠 <b>AI Summary:</b> {html.escape(ai_data["summary_1line"])}
🇮🇳 <b>India Relevant:</b> {"Yes" if ai_data["india_relevant"] else "No"}
📊 <b>Impact:</b> {ai_data["impact"]}% ({html.escape(ai_data["impact_label"])})
💹 <b>AI Sentiment:</b> {html.escape(ai_data["sentiment"])}
🤖 <b>FinBERT:</b> {html.escape(fin_data["label"])} ({fin_data["confidence"]:.2f})
🏭 <b>Sector:</b> {html.escape(ai_data["sector"])}
📝 <b>Reason:</b> {html.escape(ai_data["reason"])}
🕒 <b>Published:</b> {html.escape(published)}
🔗 {html.escape(link)}
""".strip()

# =========================
# MAIN BOT
# =========================
def run_bot():
    global seen_news

    reset_daily_counter_if_needed()

    logging.info("Fetching market feed...")
    feed = feedparser.parse(RSS_URL)
    entries = getattr(feed, "entries", [])
    logging.info("Feed fetched: %s entries", len(entries))

    sent_count = 0
    strong_buy = 0
    strong_sell = 0
    skipped_prefilter = 0
    skipped_low_impact = 0
    processed_for_ai = 0

    for entry in entries[:MAX_NEWS_PER_RUN]:
        title = clean_text(getattr(entry, "title", ""))
        summary = clean_text(getattr(entry, "summary", ""))
        link = clean_text(getattr(entry, "link", ""))
        published = clean_text(entry.get("published", "Unknown time"))

        if not title:
            continue

        news_id = make_id(title)
        if news_id in seen_news:
            continue
        seen_news.add(news_id)

        if not is_fresh(entry):
            continue

        combined_text = f"{title} {summary}"

        if not has_market_keyword(combined_text):
            skipped_prefilter += 1
            continue

        # Cheap India relevance filter BEFORE OpenAI
        if not is_india_relevant_rule(combined_text):
            skipped_prefilter += 1
            continue

        processed_for_ai += 1

        fin_data = finbert_sentiment(title)
        ai_data = ask_openai(title, summary, link)

        if not ai_data["india_relevant"] and not is_india_relevant_rule(combined_text):
            continue

        if ai_data["impact"] < MIN_IMPACT_TO_SEND:
            skipped_low_impact += 1
            continue

        msg = format_message(title, published, link, ai_data, fin_data)
        send_telegram(msg)
        sent_count += 1

        merged = merge_sentiment(ai_data["sentiment"], fin_data["label"], ai_data["impact"])
        label = final_signal(merged, ai_data["market_view"], ai_data["impact"])

        if "STRONG BUY" in label:
            strong_buy += 1
        elif "STRONG SELL" in label:
            strong_sell += 1

        logging.info("Alert sent: %s", title)
        time.sleep(1)

    if sent_count > 0:
        if strong_sell > 0:
            summary_signal = "📉 MARKET VIEW: STRONG SELL"
        elif strong_buy > 0:
            summary_signal = "📈 MARKET VIEW: STRONG BUY"
        else:
            summary_signal = "📊 MARKET VIEW: CAUTIOUS / NEWS DRIVEN"

        summary_msg = (
            f"{summary_signal}\n"
            f"Alerts: {sent_count} | Strong Buy: {strong_buy} | Strong Sell: {strong_sell}\n"
            f"AI checked: {processed_for_ai} | Low impact skipped: {skipped_low_impact}\n"
            f"OpenAI calls today: {openai_calls_today}/{OPENAI_CALLS_LIMIT_PER_DAY}"
        )
        logging.info(summary_msg.replace("\n", " | "))
        send_telegram(summary_msg)
    else:
        logging.info(
            "No qualifying news this run | prefilter skipped=%s | ai_checked=%s | low_impact=%s | openai_today=%s/%s",
            skipped_prefilter,
            processed_for_ai,
            skipped_low_impact,
            openai_calls_today,
            OPENAI_CALLS_LIMIT_PER_DAY
        )

def startup():
    mode = "OpenAI + FinBERT + Telegram + RSS" if OPENAI_ENABLED else "FinBERT + Telegram + RSS (Fallback Only)"
    msg = (
        f"🚀 <b>Hybrid AI Market News Bot Started</b>\n"
        f"Mode: {mode}\n"
        f"Polling every {POLL_MINUTES} min\n"
        f"Daily OpenAI limit: {OPENAI_CALLS_LIMIT_PER_DAY}"
    )
    logging.info(msg.replace("\n", " | "))
    send_telegram(msg)

# =========================
# START
# =========================
if __name__ == "__main__":
    startup()
    run_bot()
    schedule.every(POLL_MINUTES).minutes.do(run_bot)
    logging.info("Bot running...")

    while True:
        schedule.run_pending()
        time.sleep(5)