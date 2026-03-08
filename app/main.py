import os
import time
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import yfinance as yf
import anthropic

STOCKS_LIST = [
    {"ticker": "7203.T", "name": "トヨタ自動車", "sector": "自動車"},
    {"ticker": "6758.T", "name": "ソニーグループ", "sector": "電機・精密"},
    {"ticker": "7974.T", "name": "任天堂", "sector": "ゲーム・娯楽"},
    {"ticker": "9432.T", "name": "日本電信電話(NTT)", "sector": "通信"},
    {"ticker": "9433.T", "name": "KDDI", "sector": "通信"},
    {"ticker": "9434.T", "name": "ソフトバンク", "sector": "通信"},
    {"ticker": "9983.T", "name": "ファーストリテイリング", "sector": "小売"},
    {"ticker": "6861.T", "name": "キーエンス", "sector": "電機・精密"},
    {"ticker": "8306.T", "name": "三菱UFJフィナンシャル・グループ", "sector": "銀行"},
    {"ticker": "8031.T", "name": "三井物産", "sector": "商社"},
    {"ticker": "8001.T", "name": "伊藤忠商事", "sector": "商社"},
    {"ticker": "4063.T", "name": "信越化学工業", "sector": "化学"},
    {"ticker": "6367.T", "name": "ダイキン工業", "sector": "機械"},
    {"ticker": "8035.T", "name": "東京エレクトロン", "sector": "半導体"},
    {"ticker": "7267.T", "name": "本田技研工業", "sector": "自動車"},
    {"ticker": "6501.T", "name": "日立製作所", "sector": "電機・精密"},
    {"ticker": "3382.T", "name": "セブン&アイ・ホールディングス", "sector": "小売"},
    {"ticker": "6098.T", "name": "リクルートホールディングス", "sector": "サービス"},
    {"ticker": "4502.T", "name": "武田薬品工業", "sector": "医薬品"},
    {"ticker": "8591.T", "name": "オリックス", "sector": "金融"},
    {"ticker": "9020.T", "name": "JR東日本", "sector": "鉄道・運輸"},
    {"ticker": "2914.T", "name": "日本たばこ産業(JT)", "sector": "食品・飲料"},
    {"ticker": "6702.T", "name": "富士通", "sector": "ITサービス"},
    {"ticker": "8473.T", "name": "SBIホールディングス", "sector": "金融"},
    {"ticker": "2802.T", "name": "味の素", "sector": "食品・飲料"},
]

CACHE: dict = {}
CACHE_TTL = 3600  # 1 hour


def fetch_stock_data(ticker: str, japanese_name: str, sector: str) -> dict | None:
    now = time.time()
    if ticker in CACHE and now - CACHE[ticker]["ts"] < CACHE_TTL:
        return CACHE[ticker]["data"]

    try:
        info = yf.Ticker(ticker).info
        data = {
            "ticker": ticker,
            "name": japanese_name,
            "sector": sector,
            "price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "per": info.get("trailingPE"),
            "pbr": info.get("priceToBook"),
            "dividend_yield": info.get("dividendYield"),
            "market_cap": info.get("marketCap"),
            "week52_high": info.get("fiftyTwoWeekHigh"),
            "week52_low": info.get("fiftyTwoWeekLow"),
            "roe": info.get("returnOnEquity"),
            "revenue_growth": info.get("revenueGrowth"),
        }
        CACHE[ticker] = {"ts": now, "data": data}
        return data
    except Exception:
        return None


async def prefetch_all_stocks():
    for s in STOCKS_LIST:
        await asyncio.to_thread(fetch_stock_data, s["ticker"], s["name"], s["sector"])
        await asyncio.sleep(0.3)


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(prefetch_all_stocks())
    yield


app = FastAPI(title="日本株セレクター", lifespan=lifespan)


@app.get("/api/stocks")
async def get_stocks():
    results = []
    for s in STOCKS_LIST:
        data = await asyncio.to_thread(fetch_stock_data, s["ticker"], s["name"], s["sector"])
        if data:
            results.append(data)
    return {"stocks": results}


@app.post("/api/stocks/{ticker}/analyze")
async def analyze_stock(ticker: str):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY が設定されていません")

    stock_info = next((s for s in STOCKS_LIST if s["ticker"] == ticker), None)
    if not stock_info:
        raise HTTPException(status_code=404, detail="銘柄が見つかりません")

    data = await asyncio.to_thread(
        fetch_stock_data, stock_info["ticker"], stock_info["name"], stock_info["sector"]
    )

    def fmt(v, multiplier=1, suffix="", decimals=2):
        if v is None:
            return "データなし"
        return f"{v * multiplier:.{decimals}f}{suffix}"

    def fmt_cap(v):
        if not v:
            return "データなし"
        if v >= 1e12:
            return f"{v / 1e12:.1f}兆円"
        return f"{v / 1e8:.0f}億円"

    prompt = f"""あなたは株式投資の専門家で、投資初心者向けの解説を行うアドバイザーです。
以下の日本株について、投資初心者向けに分かりやすく解説してください。

**銘柄情報**
- 銘柄名: {stock_info['name']}
- ティッカー: {ticker}
- 業種: {stock_info['sector']}
- 現在株価: {fmt(data.get('price') if data else None, suffix='円', decimals=0)}
- PER: {fmt(data.get('per') if data else None, suffix='倍')}
- PBR: {fmt(data.get('pbr') if data else None, suffix='倍')}
- 配当利回り: {fmt(data.get('dividend_yield') if data else None, multiplier=100, suffix='%')}
- 時価総額: {fmt_cap(data.get('market_cap') if data else None)}

以下の4つの項目について、**中学生でも理解できる平易な日本語**で解説してください。マークダウン形式で回答してください。

### この会社について
この会社が何をしている会社か、私たちの日常生活との関連を交えて説明してください（3〜4文）。

### 投資指標の読み方
PER・PBR・配当利回りの数値を使って、この銘柄が割安か割高か、配当は多いかどうかを具体的に説明してください。

### 投資初心者へのポイント
この銘柄の特徴や、初心者が注目すべき点を3つ、箇条書きで挙げてください。

### 注意すべきリスク
この銘柄に投資する際のリスクを2〜3点、箇条書きで説明してください。

### 総合評価
初心者にとってこの銘柄が向いているかどうかを一言でまとめてください。

---
※ この解説は教育目的の情報提供であり、投資助言ではありません。投資の最終判断はご自身でお願いします。"""

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    return {"analysis": message.content[0].text}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
