import os

import anthropic
import pandas as pd
import streamlit as st
import yfinance as yf

# ─────────────────────────────────────────────
# 定数
# ─────────────────────────────────────────────
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

SECTORS = sorted({s["sector"] for s in STOCKS_LIST})


# ─────────────────────────────────────────────
# データ取得（プロセス全体でキャッシュ、1時間TTL）
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_stocks() -> list[dict]:
    results = []
    for s in STOCKS_LIST:
        try:
            info = yf.Ticker(s["ticker"]).info
            div = info.get("dividendYield")
            results.append(
                {
                    "ticker": s["ticker"],
                    "name": s["name"],
                    "sector": s["sector"],
                    "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                    "per": info.get("trailingPE"),
                    "pbr": info.get("priceToBook"),
                    "dividend_yield": div,
                    "market_cap": info.get("marketCap"),
                    "week52_high": info.get("fiftyTwoWeekHigh"),
                    "week52_low": info.get("fiftyTwoWeekLow"),
                }
            )
        except Exception:
            pass
    return results


# ─────────────────────────────────────────────
# ヘルパー関数
# ─────────────────────────────────────────────
def calc_score(s: dict) -> int:
    score = 0
    per, pbr, div, cap = s.get("per"), s.get("pbr"), s.get("dividend_yield"), s.get("market_cap")
    if per is not None:
        score += 2 if per < 15 else (1 if per < 25 else 0)
    if pbr is not None:
        score += 2 if pbr < 1 else (1 if pbr < 2 else 0)
    if div is not None:
        d = div * 100
        score += 2 if d >= 3 else (1 if d >= 2 else 0)
    if cap and cap > 5e12:
        score += 1
    return min(round(score / 7 * 5), 5)


def stars(n: int) -> str:
    return "★" * n + "☆" * (5 - n)


def fmt_price(v) -> str:
    return f"¥{int(v):,}" if v is not None else "-"


def fmt_cap(v) -> str:
    if not v:
        return "-"
    return f"{v / 1e12:.1f}兆円" if v >= 1e12 else f"{v / 1e8:.0f}億円"


def per_label(v) -> str:
    if v is None:
        return "データなし"
    label = "割安" if v < 15 else ("適正" if v < 25 else "割高")
    return f"{v:.1f}倍（{label}）"


def pbr_label(v) -> str:
    if v is None:
        return "データなし"
    label = "割安" if v < 1 else ("適正" if v < 2 else "割高")
    return f"{v:.2f}倍（{label}）"


def div_label(v) -> str:
    if v is None:
        return "データなし"
    p = v * 100
    label = "高配当" if p >= 3 else ("普通" if p >= 2 else "低配当")
    return f"{p:.2f}%（{label}）"


def build_prompt(s: dict) -> str:
    return f"""あなたは株式投資の専門家で、投資初心者向けの解説を行うアドバイザーです。
以下の日本株について、投資初心者向けに分かりやすく解説してください。

**銘柄情報**
- 銘柄名: {s['name']}
- ティッカー: {s['ticker']}
- 業種: {s['sector']}
- 現在株価: {fmt_price(s.get('price'))}
- PER: {per_label(s.get('per'))}
- PBR: {pbr_label(s.get('pbr'))}
- 配当利回り: {div_label(s.get('dividend_yield'))}
- 時価総額: {fmt_cap(s.get('market_cap'))}

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


def stream_analysis(s: dict):
    api_key = None
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        pass
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        yield "⚠️ ANTHROPIC_API_KEY が設定されていません。\n\nStreamlit Cloud をお使いの場合は、アプリ設定の **Secrets** に `ANTHROPIC_API_KEY = \"sk-ant-...\"` を追加してください。ローカルで実行する場合は、環境変数または `.streamlit/secrets.toml` に設定してください。"
        return

    client = anthropic.Anthropic(api_key=api_key)
    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{"role": "user", "content": build_prompt(s)}],
    ) as stream:
        for text in stream.text_stream:
            yield text


# ─────────────────────────────────────────────
# ページ設定
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="日本株セレクター",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# サイドバー
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 日本株セレクター")
    st.caption("投資初心者向け銘柄選びツール")
    st.divider()

    st.markdown("### 🔍 絞り込み条件")
    sector_filter = st.selectbox("業種", ["すべて"] + SECTORS)
    per_max = st.number_input(
        "PER 上限（倍）",
        min_value=0.0,
        value=None,
        placeholder="例: 20（空欄=条件なし）",
        format="%.1f",
    )
    pbr_max = st.number_input(
        "PBR 上限（倍）",
        min_value=0.0,
        value=None,
        placeholder="例: 2（空欄=条件なし）",
        format="%.2f",
    )
    div_min = st.number_input(
        "配当利回り 下限（%）",
        min_value=0.0,
        value=None,
        placeholder="例: 2（空欄=条件なし）",
        format="%.1f",
    )

    if st.button("🔄 条件をリセット", use_container_width=True):
        st.rerun()

    st.divider()

    with st.expander("📖 指標の読み方（初心者ガイド）"):
        st.markdown(
            """
**PER（株価収益率）**
株価 ÷ 1株あたり利益。数値が低いほど「稼ぎに比べて株が安い＝割安」とされます。
- 🟢 **15倍以下** → 割安の目安
- 🟡 **15〜25倍** → 適正水準
- 🔴 **25倍超** → 割高の目安

**PBR（株価純資産倍率）**
株価 ÷ 1株あたり純資産。会社を今すぐ解散した時の価値と比べた倍率です。
- 🟢 **1倍以下** → 解散価値より安い（割安感あり）
- 🟡 **1〜2倍** → 適正水準
- 🔴 **2倍超** → 割高の目安

**配当利回り**
年間配当金 ÷ 株価。株を持っているだけで受け取れる利息のようなものです。
- 🟢 **3%以上** → 高配当（安定収入が期待できる）
- 🟡 **2〜3%** → 普通
- ⚪ **2%未満** → 低配当
"""
        )

# ─────────────────────────────────────────────
# メインエリア：ヘッダー
# ─────────────────────────────────────────────
st.title("📈 日本株セレクター")
st.caption("投資初心者向け銘柄選びツール｜Yahoo Finance のデータを使用")

st.info(
    """
**このツールの使い方**
1. **左のサイドバー** で PER・PBR・配当利回りなどの条件を設定して銘柄を絞り込む
2. **テーブルの行をクリック** して銘柄の詳細メトリクスを表示する
3. **「AI分析を実行」ボタン** を押すと Claude AI が初心者向けにやさしく解説する
""",
    icon="💡",
)

# ─────────────────────────────────────────────
# データ読み込み
# ─────────────────────────────────────────────
with st.spinner("📡 株価データを取得中...（初回は30秒ほどかかります）"):
    all_stocks = fetch_all_stocks()

if not all_stocks:
    st.error("株価データの取得に失敗しました。しばらく待ってからページを更新してください。")
    st.stop()

# ─────────────────────────────────────────────
# フィルター適用
# ─────────────────────────────────────────────
filtered: list[dict] = all_stocks
if sector_filter != "すべて":
    filtered = [s for s in filtered if s["sector"] == sector_filter]
if per_max is not None:
    filtered = [s for s in filtered if s.get("per") is not None and s["per"] <= per_max]
if pbr_max is not None:
    filtered = [s for s in filtered if s.get("pbr") is not None and s["pbr"] <= pbr_max]
if div_min is not None:
    filtered = [
        s
        for s in filtered
        if s.get("dividend_yield") is not None and s["dividend_yield"] * 100 >= div_min
    ]

st.sidebar.metric("表示中の銘柄数", f"{len(filtered)} 件 / 全{len(all_stocks)}件")

if not filtered:
    st.warning("⚠️ 条件に合う銘柄が見つかりませんでした。サイドバーの条件を緩めてください。")
    st.stop()

# ─────────────────────────────────────────────
# 銘柄テーブル
# ─────────────────────────────────────────────
st.subheader(f"銘柄一覧（{len(filtered)}件）")
st.caption("行をクリックすると詳細が表示されます")

rows = []
for s in filtered:
    div = s.get("dividend_yield")
    rows.append(
        {
            "銘柄名": s["name"],
            "業種": s["sector"],
            "株価（円）": s.get("price"),
            "PER（倍）": s.get("per"),
            "PBR（倍）": s.get("pbr"),
            "配当利回り（%）": div * 100 if div is not None else None,
            "初心者おすすめ度": stars(calc_score(s)),
        }
    )

df = pd.DataFrame(rows)

event = st.dataframe(
    df,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        "株価（円）": st.column_config.NumberColumn(
            "株価（円）", format="¥%,.0f", help="直近の終値"
        ),
        "PER（倍）": st.column_config.NumberColumn(
            "PER（倍）", format="%.1f", help="低いほど割安。15倍以下が目安。"
        ),
        "PBR（倍）": st.column_config.NumberColumn(
            "PBR（倍）", format="%.2f", help="1倍以下が割安の目安。"
        ),
        "配当利回り（%）": st.column_config.NumberColumn(
            "配当利回り（%）", format="%.2f", help="3%以上が高配当の目安。"
        ),
        "初心者おすすめ度": st.column_config.TextColumn(
            "おすすめ度 ★", help="PER・PBR・配当利回り・時価総額から自動算出（★5が最高）"
        ),
    },
)

# ─────────────────────────────────────────────
# 銘柄詳細パネル（行選択時）
# ─────────────────────────────────────────────
selected_rows = event.selection.rows
if not selected_rows:
    st.caption("↑ 行をクリックすると詳細が表示されます")
    st.stop()

stock = filtered[selected_rows[0]]
ticker = stock["ticker"]
score = calc_score(stock)

st.divider()
st.subheader(f"📊 {stock['name']}（{ticker}）")
st.caption(f"業種: {stock['sector']}　｜　初心者おすすめ度: {stars(score)} ({score}/5)")

# サマリー指標（4列）
c1, c2, c3, c4 = st.columns(4)
c1.metric("現在株価", fmt_price(stock.get("price")))
c2.metric("時価総額", fmt_cap(stock.get("market_cap")))
c3.metric("52週高値", fmt_price(stock.get("week52_high")))
c4.metric("52週安値", fmt_price(stock.get("week52_low")))

# 詳細指標（3列）
col_per, col_pbr, col_div = st.columns(3)

with col_per:
    v = stock.get("per")
    icon = "🟢" if v and v < 15 else ("🟡" if v and v < 25 else "🔴")
    st.metric("PER（株価収益率）", per_label(v))
    st.caption(f"{icon} 低いほど割安。15倍以下が目安。")

with col_pbr:
    v = stock.get("pbr")
    icon = "🟢" if v and v < 1 else ("🟡" if v and v < 2 else "🔴")
    st.metric("PBR（株価純資産倍率）", pbr_label(v))
    st.caption(f"{icon} 1倍以下は解散価値より安い。")

with col_div:
    v = stock.get("dividend_yield")
    p = v * 100 if v else 0
    icon = "🟢" if p >= 3 else ("🟡" if p >= 2 else "⚪")
    st.metric("配当利回り", div_label(v))
    st.caption(f"{icon} 3%以上が高配当とされる。")

# ─────────────────────────────────────────────
# AI 分析セクション
# ─────────────────────────────────────────────
st.divider()
st.markdown("#### 🤖 AI 分析・解説（Claude AI）")

analysis_key = f"analysis_{ticker}"

if st.button("✨ AI分析を実行", type="primary", key=f"btn_{ticker}"):
    result = st.write_stream(stream_analysis(stock))
    st.session_state[analysis_key] = result
    st.warning(
        "⚠️ この分析は AI によるものです。投資の最終判断はご自身の責任でお願いします。",
        icon="⚠️",
    )
elif analysis_key in st.session_state:
    st.markdown(st.session_state[analysis_key])
    st.warning(
        "⚠️ この分析は AI によるものです。投資の最終判断はご自身の責任でお願いします。",
        icon="⚠️",
    )
else:
    st.caption(
        "「AI分析を実行」を押すと、Claude AI がこの銘柄を初心者向けにやさしく解説します。"
    )
