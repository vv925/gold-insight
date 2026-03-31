import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import time
from datetime import datetime

st.set_page_config(page_title="Gold Insight", page_icon="✨", layout="wide")

# =========================
# Sell Trigger Config
# =========================
SELL_CONFIG = {
    "rmb_key_level_1": 1030,
    "rmb_key_level_2": 1050,
    "rmb_key_level_3": 1080,
    "volume_expand": 1.20,
    "volume_strong": 1.50,
    "volume_shrink": 0.90,
    "main_position_defense": 1000,
    "main_position_risk": 980,
    "trading_sell_small": "卖出交易仓 10%-15%",
    "trading_sell_medium": "卖出交易仓 20%-25%",
    "trading_sell_large": "卖出交易仓 30%-40%",
    "main_position_action": "主力仓暂不主动卖出，仅做风控观察"
}

# =========================
# Data Loading with Retry
# =========================
@st.cache_data(ttl=1800)
def load_data(period="6mo"):
    tickers = {
        "Gold_USD_oz": "GC=F",
        "Gold_ETF": "GLD",
        "DXY": "DX-Y.NYB",
        "UST10Y": "^TNX",
        "Oil": "CL=F",
        "USDCNY": "CNY=X"
    }

    frames = []
    failed_sources = {}

    for name, ticker in tickers.items():
        raw_df = pd.DataFrame()
        last_error = None

        for attempt in range(3):
            try:
                raw_df = yf.download(
                    ticker,
                    period=period,
                    progress=False,
                    auto_adjust=False,
                    threads=False
                )

                if not raw_df.empty:
                    break

                last_error = f"{ticker}: empty result (attempt {attempt + 1})"
                time.sleep(1.2)

            except Exception as e:
                last_error = f"{ticker}: {e} (attempt {attempt + 1})"
                time.sleep(1.2)

        if raw_df.empty:
            failed_sources[name] = last_error or f"{ticker}: failed"
            continue

        raw_df = raw_df.reset_index()

        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = [c[0] for c in raw_df.columns]

        keep_cols = ["Date", "Close"]
        if "Volume" in raw_df.columns:
            keep_cols.append("Volume")

        if not all(col in raw_df.columns for col in keep_cols):
            failed_sources[name] = f"{ticker}: missing required columns"
            continue

        df = raw_df[keep_cols].copy()
        df = df.rename(columns={"Close": name})

        if "Volume" in df.columns:
            df = df.rename(columns={"Volume": f"{name}_Volume"})

        frames.append(df)

    if not frames:
        return pd.DataFrame(), failed_sources

    merged = frames[0]
    for frame in frames[1:]:
        merged = pd.merge(merged, frame, on="Date", how="outer")

    merged = merged.sort_values("Date")

    if "Gold_USD_oz" in merged.columns and "USDCNY" in merged.columns:
        merged["Gold_RMB_g"] = merged["Gold_USD_oz"] * merged["USDCNY"] / 31.1035

    return merged, failed_sources

# =========================
# Helpers
# =========================
def latest_valid(series):
    s = series.dropna()
    return s.iloc[-1] if not s.empty else np.nan

def pct_change_days(series, days=20):
    s = series.dropna()
    if len(s) <= days:
        return np.nan
    return (s.iloc[-1] / s.iloc[-1 - days] - 1) * 100

def moving_avg(series, window=20):
    s = series.dropna()
    if len(s) < window:
        return np.nan
    return s.rolling(window).mean().iloc[-1]

def prev(df, col):
    if col not in df.columns:
        return None
    s = df[col].dropna()
    return s.iloc[-2] if len(s) > 1 else None

def pct(lat, pre):
    if lat is not None and pre is not None and pre != 0 and not pd.isna(lat) and not pd.isna(pre):
        return (lat / pre - 1) * 100
    return None

def fmt(x, d=2):
    if x is None or pd.isna(x):
        return "N/A"
    return f"{x:,.{d}f}"

def latest(df, col):
    if col not in df.columns:
        return None
    s = df[col].dropna()
    return s.iloc[-1] if len(s) > 0 else None

# =========================
# Regime Score
# =========================
def get_regime_score(df):
    gold = df["Gold_USD_oz"].dropna() if "Gold_USD_oz" in df.columns else pd.Series(dtype=float)
    dxy = df["DXY"].dropna() if "DXY" in df.columns else pd.Series(dtype=float)
    ust10y = df["UST10Y"].dropna() if "UST10Y" in df.columns else pd.Series(dtype=float)
    oil = df["Oil"].dropna() if "Oil" in df.columns else pd.Series(dtype=float)
    usdcny = df["USDCNY"].dropna() if "USDCNY" in df.columns else pd.Series(dtype=float)

    gold_now = latest_valid(gold)
    dxy_now = latest_valid(dxy)
    ust_now = latest_valid(ust10y)
    oil_now = latest_valid(oil)
    fx_now = latest_valid(usdcny)

    gold_ma20 = moving_avg(gold, 20)
    gold_ma60 = moving_avg(gold, 60)
    dxy_ma20 = moving_avg(dxy, 20)
    ust_ma20 = moving_avg(ust10y, 20)
    oil_ma20 = moving_avg(oil, 20)

    gold_chg20 = pct_change_days(gold, 20)
    dxy_chg20 = pct_change_days(dxy, 20)
    ust_chg20 = pct_change_days(ust10y, 20)
    oil_chg20 = pct_change_days(oil, 20)

    gold_trend_score = 0

    if pd.notna(gold_now) and pd.notna(gold_ma20) and gold_now > gold_ma20:
        gold_trend_score += 1
    else:
        gold_trend_score -= 1

    if pd.notna(gold_ma20) and pd.notna(gold_ma60) and gold_ma20 > gold_ma60:
        gold_trend_score += 1
    else:
        gold_trend_score -= 1

    if pd.notna(gold_chg20):
        if gold_chg20 > 3:
            gold_trend_score += 1
        elif gold_chg20 < -3:
            gold_trend_score -= 1

    macro_score = 0

    if pd.notna(dxy_now) and pd.notna(dxy_ma20):
        macro_score += 1 if dxy_now < dxy_ma20 else -1

    if pd.notna(ust_now) and pd.notna(ust_ma20):
        macro_score += 1 if ust_now < ust_ma20 else -1

    if pd.notna(oil_now) and pd.notna(oil_ma20):
        macro_score += 0.5 if oil_now > oil_ma20 else -0.5

    total_score = gold_trend_score + macro_score

    if gold_trend_score >= 2 and macro_score >= 0:
        regime = "牛市"
    elif gold_trend_score >= 0 and macro_score < 0:
        regime = "挤泡沫"
    elif gold_trend_score <= -2 and macro_score <= -1:
        regime = "熊市"
    else:
        regime = "震荡"

    gold_rmb_g = np.nan
    if pd.notna(gold_now) and pd.notna(fx_now):
        gold_rmb_g = gold_now * fx_now / 31.1035

    action = "观望"
    trigger_text = "暂无明显触发"

    if pd.notna(gold_rmb_g):
        if regime == "牛市":
            if gold_rmb_g >= 1250:
                action = "减仓"
                trigger_text = "价格进入高位止盈区（1250+ RMB/g）"
            elif gold_rmb_g <= 1080:
                action = "小买"
                trigger_text = "价格回到第一试探买入区（1080附近）"
            else:
                trigger_text = "牛市中，价格未到关键买卖区"

        elif regime == "挤泡沫":
            if gold_rmb_g <= 1000:
                action = "加仓"
                trigger_text = "挤泡沫阶段 + 价格进入主力加仓区（1000附近）"
            elif gold_rmb_g <= 1080:
                action = "小买"
                trigger_text = "挤泡沫阶段 + 价格进入试探区（1080附近）"
            elif gold_rmb_g >= 1250:
                action = "减仓"
                trigger_text = "反弹至高位，可分批减仓"
            else:
                trigger_text = "挤泡沫阶段，等待更好性价比位置"

        elif regime == "熊市":
            if gold_rmb_g >= 1200:
                action = "减仓"
                trigger_text = "熊市框架下，反弹优先减仓"
            elif gold_rmb_g <= 950:
                action = "小买"
                trigger_text = "极端杀跌区，仅允许非常小仓位试探"
            else:
                trigger_text = "熊市中以等待为主"

        else:
            if gold_rmb_g <= 1080:
                action = "小买"
                trigger_text = "震荡市下沿，可轻仓试探"
            elif gold_rmb_g >= 1250:
                action = "减仓"
                trigger_text = "震荡市上沿，可分批减仓"
            else:
                trigger_text = "震荡区间中部，等待"

    return {
        "gold_now": gold_now,
        "gold_rmb_g": gold_rmb_g,
        "dxy_now": dxy_now,
        "ust_now": ust_now,
        "oil_now": oil_now,
        "gold_chg20": gold_chg20,
        "dxy_chg20": dxy_chg20,
        "ust_chg20": ust_chg20,
        "oil_chg20": oil_chg20,
        "gold_trend_score": gold_trend_score,
        "macro_score": macro_score,
        "total_score": total_score,
        "regime": regime,
        "action": action,
        "trigger_text": trigger_text,
    }

# =========================
# Volume Analysis
# =========================
def volume_analysis(df):
    if "Gold_ETF_Volume" not in df.columns:
        return "无数据", "", None

    vol = df["Gold_ETF_Volume"].dropna()
    if len(vol) < 20:
        return "无数据", "", None

    latest_vol = vol.iloc[-1]
    avg = vol.tail(20).mean()
    ratio = latest_vol / avg if avg and not pd.isna(avg) else None

    if ratio is None:
        return "无数据", "", None
    elif ratio > 1.5:
        return "放量", f"{ratio:.2f}x 均量", ratio
    elif ratio < 0.7:
        return "缩量", f"{ratio:.2f}x 均量", ratio
    else:
        return "正常", f"{ratio:.2f}x 均量", ratio

def volume_price(df):
    if "Gold_USD_oz" not in df.columns or "Gold_ETF_Volume" not in df.columns:
        return "无信号"

    price = df["Gold_USD_oz"].dropna()
    vol = df["Gold_ETF_Volume"].dropna()

    if len(price) < 2 or len(vol) < 20:
        return "无信号"

    change = price.iloc[-1] - price.iloc[-2]
    ratio = vol.iloc[-1] / vol.tail(20).mean()

    if change > 0 and ratio > 1.3:
        return "放量上涨（强势）"
    elif change > 0:
        return "缩量上涨（谨慎）"
    elif change < 0 and ratio > 1.3:
        return "放量下跌（偏空）"
    else:
        return "正常回调"

# =========================
# Sell Signal Engine
# =========================
def evaluate_sell_signal(gold_rmb, phase, total_score, volume_ratio, price_change_pct):
    cfg = SELL_CONFIG

    if gold_rmb is None or pd.isna(gold_rmb):
        return {
            "sell_action": "无数据",
            "sell_level": "无法判断",
            "sell_reason": "当前黄金主价格源未成功获取，暂时无法生成卖出信号。",
            "sell_target": "N/A",
            "position_scope": "N/A",
            "signal_tag": "无数据"
        }

    if volume_ratio is None or pd.isna(volume_ratio):
        volume_ratio = 1.0
    if price_change_pct is None or pd.isna(price_change_pct):
        price_change_pct = 0

    sell_action = "暂不卖出"
    sell_level = "无卖点确认"
    sell_reason = "当前未出现放量上冲或冲高失败信号，继续观察。"
    sell_target = "0%"
    position_scope = "以交易仓为主"
    signal_tag = "等待"

    if price_change_pct > 0 and volume_ratio < cfg["volume_shrink"]:
        sell_action = "暂不卖出"
        sell_level = "缩量上涨"
        sell_reason = "上涨但未放量，更多像弱反弹或空头回补，不构成理想卖点。"
        sell_target = "0%"
        position_scope = "不动主力仓，不动交易仓"
        signal_tag = "观望"

    elif (
        price_change_pct > 0
        and volume_ratio >= cfg["volume_expand"]
        and gold_rmb >= cfg["rmb_key_level_1"]
    ):
        sell_action = cfg["trading_sell_small"]
        sell_level = "第一档卖点"
        sell_reason = f"金价反弹至 {cfg['rmb_key_level_1']} 以上且出现放量，适合先兑现一小部分交易仓。"
        sell_target = "10%-15%"
        position_scope = "仅交易仓"
        signal_tag = "轻卖"

    if (
        price_change_pct > 0
        and volume_ratio >= cfg["volume_expand"]
        and gold_rmb >= cfg["rmb_key_level_2"]
    ):
        sell_action = cfg["trading_sell_medium"]
        sell_level = "第二档卖点"
        sell_reason = f"金价反弹至 {cfg['rmb_key_level_2']} 以上并放量，反弹质量更高，可继续分批兑现交易仓。"
        sell_target = "20%-25%"
        position_scope = "仅交易仓"
        signal_tag = "分批卖"

    if (
        price_change_pct > 0
        and volume_ratio >= cfg["volume_strong"]
        and gold_rmb >= cfg["rmb_key_level_3"]
    ):
        sell_action = cfg["trading_sell_large"]
        sell_level = "强反弹卖点"
        sell_reason = f"金价到达 {cfg['rmb_key_level_3']} 附近且明显放量，属于较理想的反弹兑现窗口。"
        sell_target = "30%-40%"
        position_scope = "仅交易仓"
        signal_tag = "强卖"

    if phase == "熊市" and price_change_pct < 0 and volume_ratio < cfg["volume_expand"]:
        sell_action = "不追卖"
        sell_level = "低位弱势区"
        sell_reason = "处于偏弱阶段但未见明确反弹卖点，此时继续追卖容易卖在低位。"
        sell_target = "0%"
        position_scope = "主力仓不动，交易仓等待反弹"
        signal_tag = "等待反弹"

    if gold_rmb < cfg["main_position_defense"] and total_score <= -6:
        sell_action = "主力仓进入风控观察"
        sell_level = "主仓防守区"
        sell_reason = f"金价跌破 {cfg['main_position_defense']} 且总评分明显偏空，需关注主仓风险，但暂不建议机械性砍仓。"
        sell_target = "主仓暂不自动卖出"
        position_scope = "主力仓 + 交易仓整体复核"
        signal_tag = "风控观察"

    if gold_rmb < cfg["main_position_risk"] and total_score <= -8:
        sell_action = "主力仓风险警报"
        sell_level = "主仓高风险区"
        sell_reason = f"金价跌破 {cfg['main_position_risk']} 且评分极弱，说明趋势转坏，需要重新评估主力仓位防守策略。"
        sell_target = "人工复核后决定"
        position_scope = "主力仓重点评估"
        signal_tag = "高风险"

    return {
        "sell_action": sell_action,
        "sell_level": sell_level,
        "sell_reason": sell_reason,
        "sell_target": sell_target,
        "position_scope": position_scope,
        "signal_tag": signal_tag
    }

def render_sell_signal_box(signal):
    tag = signal["signal_tag"]

    if tag in ["强卖", "分批卖", "轻卖"]:
        st.success(f"卖出建议：{signal['sell_action']}｜目标：{signal['sell_target']}｜{signal['sell_reason']}")
    elif tag in ["风控观察", "高风险"]:
        st.warning(f"风险提示：{signal['sell_action']}｜{signal['sell_reason']}")
    else:
        st.info(f"当前状态：{signal['sell_action']}｜{signal['sell_reason']}")

def metric_tuple(label, val, prev):
    delta = None if val is None or prev is None or pd.isna(val) or pd.isna(prev) else val - prev
    pct_val = pct(val, prev)
    delta_str = None if delta is None or pct_val is None else f"{delta:,.2f} ({pct_val:.2f}%)"
    return label, fmt(val), delta_str

# =========================
# UI Style
# =========================
st.markdown("""
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    padding-left: 1.2rem;
    padding-right: 1.2rem;
}
.main-title {
    font-size: 2.3rem;
    font-weight: 700;
    margin-bottom: 0.15rem;
}
.sub-title {
    font-size: 1rem;
    color: #888;
    margin-bottom: 1.2rem;
}
.small-note {
    color: #777;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Sidebar
# =========================
st.sidebar.title("设置")
mobile_mode = st.sidebar.toggle("手机精简模式", value=True)
show_debug = st.sidebar.toggle("显示数据源状态", value=False)
period = st.sidebar.selectbox("数据区间", ["3mo", "6mo", "1y"], index=1)

# =========================
# Main Data Prep
# =========================
df, failed_sources = load_data(period=period)

if df.empty:
    st.error("未能成功加载市场数据，请稍后刷新重试。")
    st.stop()

signal = get_regime_score(df)

gold = latest(df, "Gold_USD_oz")
gold_prev = prev(df, "Gold_USD_oz")

dxy = latest(df, "DXY")
dxy_prev = prev(df, "DXY")

ust = latest(df, "UST10Y")
ust_prev = prev(df, "UST10Y")

oil = latest(df, "Oil")
oil_prev = prev(df, "Oil")

rmb = latest(df, "Gold_RMB_g")
rmb_prev = prev(df, "Gold_RMB_g")

phase = signal["regime"]
total_score = signal["total_score"]
gold_rmb_change_pct = pct(rmb, rmb_prev)

vol_state, vol_desc, volume_ratio = volume_analysis(df)

sell_signal = evaluate_sell_signal(
    gold_rmb=rmb,
    phase=phase,
    total_score=total_score,
    volume_ratio=volume_ratio,
    price_change_pct=gold_rmb_change_pct
)

# =========================
# Header
# =========================
st.markdown('<div class="main-title">Gold Insight</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">黄金市场观察与交易判断面板 · Cloud-ready · Mobile-friendly</div>',
    unsafe_allow_html=True
)

if ("Gold_USD_oz" not in df.columns) or df["Gold_USD_oz"].dropna().empty:
    st.warning("黄金主价格源本次未成功获取，其他宏观指标仍正常显示。可稍后刷新重试。")

if show_debug and failed_sources:
    with st.expander("数据源状态 / Debug"):
        st.json(failed_sources)

# =========================
# Top Decision
# =========================
st.markdown("### 今日自动判断")

if mobile_mode:
    a1, a2 = st.columns(2)
    a3, a4 = st.columns(2)
else:
    a1, a2, a3, a4 = st.columns(4)

with a1:
    st.metric("当前阶段", signal["regime"])

with a2:
    st.metric("今日建议", signal["action"])

with a3:
    if pd.notna(signal["gold_rmb_g"]):
        st.metric("估算金价(RMB/g)", f"{signal['gold_rmb_g']:.1f}")
    else:
        st.metric("估算金价(RMB/g)", "N/A")

with a4:
    st.metric("总评分", f"{signal['total_score']:.1f}")

st.info(f"触发逻辑：{signal['trigger_text']}")

st.divider()

# =========================
# Core Metrics
# =========================
st.markdown("## 核心市场指标")

if mobile_mode:
    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)

    r1c1.metric(*metric_tuple("黄金 USD/oz", gold, gold_prev))
    r1c2.metric(*metric_tuple("黄金 RMB/g", rmb, rmb_prev))
    r2c1.metric(*metric_tuple("DXY（美元指数）", dxy, dxy_prev))
    r2c2.metric(*metric_tuple("UST10Y（10年利率）", ust, ust_prev))

    st.metric(*metric_tuple("Oil", oil, oil_prev))
else:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(*metric_tuple("黄金 USD/oz", gold, gold_prev))
    c2.metric(*metric_tuple("黄金 RMB/g", rmb, rmb_prev))
    c3.metric(*metric_tuple("DXY（美元指数）", dxy, dxy_prev))
    c4.metric(*metric_tuple("UST10Y（10年利率）", ust, ust_prev))
    c5.metric(*metric_tuple("Oil", oil, oil_prev))

st.divider()

# =========================
# Sell Guidance
# =========================
st.markdown("## 今日卖出判断")

if mobile_mode:
    s1, s2 = st.columns(2)
    with s1:
        st.metric("卖出建议", sell_signal["sell_action"])
    with s2:
        st.metric("卖出范围", sell_signal["sell_target"])
    st.metric("适用仓位", sell_signal["position_scope"])
else:
    s1, s2, s3 = st.columns(3)
    s1.metric("卖出建议", sell_signal["sell_action"])
    s2.metric("卖出范围", sell_signal["sell_target"])
    s3.metric("适用仓位", sell_signal["position_scope"])

st.markdown(f"**卖点级别：** {sell_signal['sell_level']}")
st.markdown(f"**信号标签：** {sell_signal['signal_tag']}")
st.info(f"**触发原因：** {sell_signal['sell_reason']}")

render_sell_signal_box(sell_signal)

st.divider()

# =========================
# Volume Analysis
# =========================
st.markdown("## 成交量分析")

if vol_state == "放量":
    st.success(vol_state)
elif vol_state == "缩量":
    st.warning(vol_state)
else:
    st.info(vol_state)

if vol_desc:
    st.write(vol_desc)

st.write("量价关系：", volume_price(df))

st.divider()

# =========================
# Charts
# =========================
if mobile_mode:
    st.markdown("## 金价走势")
    if "Gold_USD_oz" in df.columns:
        st.line_chart(df.set_index("Date")[["Gold_USD_oz"]])

    st.markdown("## 宏观信号")
    macro_cols = [col for col in ["DXY", "UST10Y", "Oil"] if col in df.columns]
    if macro_cols:
        st.line_chart(df.set_index("Date")[macro_cols])
else:
    tab1, tab2 = st.tabs(["Gold Trend", "Macro Signals"])

    with tab1:
        st.markdown("**Gold Price Trend**")
        if "Gold_USD_oz" in df.columns:
            st.line_chart(df.set_index("Date")[["Gold_USD_oz"]])

        if "Gold_ETF_Volume" in df.columns:
            st.markdown("**Gold ETF Volume**")
            st.bar_chart(df.set_index("Date")[["Gold_ETF_Volume"]])

    with tab2:
        st.markdown("**Macro Signal Trend**")
        macro_cols = [col for col in ["DXY", "UST10Y", "Oil"] if col in df.columns]
        if macro_cols:
            st.line_chart(df.set_index("Date")[macro_cols])

st.markdown(
    f'<div class="small-note">Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>',
    unsafe_allow_html=True
)
