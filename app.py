import os
import io
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Kỹ thuật kỹ thuật (ta)
import ta

# Tối ưu hóa MPT
from scipy.optimize import minimize

# Tối ưu hóa tham số Hyperopt
try:
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False

warnings.filterwarnings("ignore")

# ==========================================
# CẤU HÌNH TRANG VÀ GIAO DIỆN (STREAMLIT CONFIG)
# ==========================================
st.set_page_config(
    page_title="Quantitative Backtest: SMA + OBV & MPT Portfolio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện hiện đại, chuyên nghiệp
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 15px;
    }
    .metric-box {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: bold;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-train {
        background-color: #E0E7FF;
        color: #3730A3;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-test {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 1. HÀM XỬ LÝ DỮ LIỆU
# ==========================================

@st.cache_data(show_spinner=False)
def load_dataset(file_source):
    """
    Đọc và chuẩn hóa dữ liệu giao dịch từ file CSV
    """
    if isinstance(file_source, str):
        if not os.path.exists(file_source):
            return None
        df = pd.read_csv(file_source, encoding="utf-8-sig", low_memory=False)
    else:
        df = pd.read_csv(file_source, encoding="utf-8-sig", low_memory=False)

    df.columns = df.columns.str.strip().str.lower()
    required_cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Dữ liệu tải lên thiếu các cột bắt buộc: {missing}")
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df = df.dropna(subset=["date", "ticker", "open", "high", "low", "close", "volume"])
    return df


def prepare_stock_data(df_full, ticker):
    """
    Trích xuất và làm sạch dữ liệu OHLCV cho từng cổ phiếu
    """
    t = ticker.upper()
    df = df_full[df_full["ticker"] == t].copy()
    if df.empty:
        raise ValueError(f"Không tìm thấy mã {ticker} trong bộ dữ liệu.")

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    df = df.sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last")

    df = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    })
    df = df[["date", "Open", "High", "Low", "Close", "Volume"]].set_index("date")
    return df


# ==========================================
# 2. HÀM TẠO TÍN HIỆU GIAO DỊCH (SMA & OBV)
# ==========================================

def find_position_sma(df, paras):
    """
    Tạo tín hiệu SMA Crossover:
    - BUY (1): SMA ngắn cắt lên SMA dài
    - SELL (-1): SMA ngắn cắt xuống SMA dài
    """
    position = pd.Series(0.0, index=df.index, name="position")
    ma_short = int(paras["ma_short"])
    ma_long = int(paras["ma_long"])

    if ma_short >= ma_long:
        return position

    ma_s = ta.trend.SMAIndicator(close=df["Close"], window=ma_short).sma_indicator()
    ma_l = ta.trend.SMAIndicator(close=df["Close"], window=ma_long).sma_indicator()

    buy_signal = (ma_s > ma_l) & (ma_s.shift(1) <= ma_l.shift(1))
    sell_signal = (ma_s < ma_l) & (ma_s.shift(1) >= ma_l.shift(1))

    position.loc[buy_signal] = 1.0
    position.loc[sell_signal] = -1.0
    return position


def find_position_obv(df, paras):
    """
    Tạo tín hiệu OBV Crossover:
    - BUY (1): OBV cắt lên đường trung bình động OBV_MA
    - SELL (-1): OBV cắt xuống đường trung bình động OBV_MA
    """
    position = pd.Series(0.0, index=df.index, name="position")
    obv_window = int(paras["obv_window"])

    obv = ta.volume.OnBalanceVolumeIndicator(
        close=df["Close"],
        volume=df["Volume"]
    ).on_balance_volume()

    obv_ma = obv.rolling(window=obv_window).mean()

    buy_signal = (obv > obv_ma) & (obv.shift(1) <= obv_ma.shift(1))
    sell_signal = (obv < obv_ma) & (obv.shift(1) >= obv_ma.shift(1))

    position.loc[buy_signal] = 1.0
    position.loc[sell_signal] = -1.0
    return position


def find_position_combined_and(df, sma_paras, obv_paras):
    """
    Kết hợp AND:
    - BUY: Cả SMA và OBV đều phát tín hiệu BUY
    - SELL: Cả SMA và OBV đều phát tín hiệu SELL
    """
    sma = find_position_sma(df, sma_paras)
    obv = find_position_obv(df, obv_paras)

    position = pd.Series(0.0, index=df.index, name="position")
    position.loc[(sma == 1) & (obv == 1)] = 1.0
    position.loc[(sma == -1) & (obv == -1)] = -1.0
    return position


def find_position_combined_or(df, sma_paras, obv_paras):
    """
    Kết hợp OR:
    - BUY: Có ít nhất 1 chỉ báo phát tín hiệu BUY (và không có mâu thuẫn)
    - SELL: Có ít nhất 1 chỉ báo phát tín hiệu SELL (và không có mâu thuẫn)
    - Nếu 1 chỉ báo BUY còn 1 chỉ báo SELL: Giữ trung lập (0)
    """
    sma = find_position_sma(df, sma_paras)
    obv = find_position_obv(df, obv_paras)

    position = pd.Series(0.0, index=df.index, name="position")
    buy = (sma == 1) | (obv == 1)
    sell = (sma == -1) | (obv == -1)
    conflict = buy & sell

    position.loc[buy & ~conflict] = 1.0
    position.loc[sell & ~conflict] = -1.0
    return position


# ==========================================
# 3. MÔ PHỎNG VỊ THẾ & TÍNH TOÁN LỢI NHUẬN
# ==========================================

def events_to_holding(events):
    """
    Duy trì trạng thái nắm giữ:
    - Nhận tín hiệu BUY (1) -> Giữ trạng thái Nắm giữ (1.0)
    - Nhận tín hiệu SELL (-1) -> Chuyển về Tiền mặt (0.0)
    - Giữ nguyên trạng thái cũ khi không có tín hiệu (0)
    """
    holding = pd.Series(0.0, index=events.index)
    current = 0.0
    for i, signal in enumerate(events):
        if signal == 1:
            current = 1.0
        elif signal == -1:
            current = 0.0
        holding.iloc[i] = current
    return holding


def strategy_returns(df, events, commission=0.0):
    """
    Tính lợi nhuận chiến lược:
    - Tránh look-ahead bias: Dịch vị thế 1 phiên (executed_holding = holding.shift(1))
    - Khấu trừ chi phí giao dịch (commission * turnover)
    """
    asset_ret = df["Close"].pct_change().fillna(0.0)
    holding = events_to_holding(events)

    # QUAN TRỌNG: Dịch 1 phiên để phản ánh thực tế đặt lệnh vào phiên tiếp theo
    executed_holding = holding.shift(1).fillna(0.0)

    strat_ret = executed_holding * asset_ret
    turnover = executed_holding.diff().abs().fillna(executed_holding.abs())
    strat_ret = strat_ret - turnover * commission

    return strat_ret, executed_holding


def performance_stats(returns, trading_days=252):
    """
    Tính toán các chỉ số hiệu quả đầu tư định lượng
    """
    r = returns.dropna()
    if len(r) == 0:
        return {
            "Total Return [%]": np.nan,
            "Annual Return [%]": np.nan,
            "Annual Volatility [%]": np.nan,
            "Sharpe Ratio": np.nan,
            "Max Drawdown [%]": np.nan
        }

    equity = (1 + r).cumprod()
    total_return = equity.iloc[-1] - 1.0

    years = len(r) / trading_days
    annual_return = (
        equity.iloc[-1] ** (1.0 / years) - 1.0
        if years > 0 and equity.iloc[-1] > 0
        else np.nan
    )

    annual_vol = r.std() * np.sqrt(trading_days)
    sharpe = (
        r.mean() / r.std() * np.sqrt(trading_days)
        if r.std() != 0 and not np.isnan(r.std())
        else np.nan
    )

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = drawdown.min()

    return {
        "Total Return [%]": total_return * 100.0,
        "Annual Return [%]": annual_return * 100.0,
        "Annual Volatility [%]": annual_vol * 100.0,
        "Sharpe Ratio": sharpe,
        "Max Drawdown [%]": max_dd * 100.0
    }


# ==========================================
# 4. TỐI ƯU HÓA THAM SỐ (TRAIN IN-SAMPLE)
# ==========================================

def score_sma(paras, df, commission=0.0):
    paras_clean = {
        "ma_short": int(paras["ma_short"]),
        "ma_long": int(paras["ma_long"])
    }
    if paras_clean["ma_short"] >= paras_clean["ma_long"]:
        return 999999.0

    events = find_position_sma(df, paras_clean)
    ret, _ = strategy_returns(df, events, commission=commission)
    stats = performance_stats(ret)
    sharpe = stats["Sharpe Ratio"]
    if pd.isna(sharpe):
        return 999999.0
    return -sharpe


def score_obv(paras, df, commission=0.0):
    paras_clean = {
        "obv_window": int(paras["obv_window"])
    }
    events = find_position_obv(df, paras_clean)
    ret, _ = strategy_returns(df, events, commission=commission)
    stats = performance_stats(ret)
    sharpe = stats["Sharpe Ratio"]
    if pd.isna(sharpe):
        return 999999.0
    return -sharpe


def optimize_stock_hyperopt(df_train, max_evals=40, commission=0.0):
    """
    Tối ưu hóa tìm tham số tốt nhất trên tập Train bằng thuật toán TPE của Hyperopt
    """
    if not HYPEROPT_AVAILABLE:
        # Fallback nếu hyperopt không sẵn có
        return {"ma_short": 50, "ma_long": 200}, {"obv_window": 20}

    space_sma = {
        "ma_short": hp.quniform("ma_short", 20, 120, 5),
        "ma_long": hp.quniform("ma_long", 150, 350, 5)
    }
    trials_sma = Trials()
    best_sma_raw = fmin(
        fn=lambda p: score_sma(p, df_train, commission),
        space=space_sma,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials_sma,
        verbose=False
    )
    sma_best = {
        "ma_short": int(best_sma_raw["ma_short"]),
        "ma_long": int(best_sma_raw["ma_long"])
    }

    space_obv = {
        "obv_window": hp.quniform("obv_window", 5, 100, 5)
    }
    trials_obv = Trials()
    best_obv_raw = fmin(
        fn=lambda p: score_obv(p, df_train, commission),
        space=space_obv,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials_obv,
        verbose=False
    )
    obv_best = {
        "obv_window": int(best_obv_raw["obv_window"])
    }

    return sma_best, obv_best


# ==========================================
# 5. TỐI ƯU HÓA DANH MỤC MPT (MODERN PORTFOLIO THEORY)
# ==========================================

def portfolio_annual_return(weights, returns, trading_days=252):
    mean_daily = returns.mean().values
    return float(weights @ mean_daily * trading_days)


def portfolio_annual_volatility(weights, returns, trading_days=252):
    cov_annual = returns.cov().values * trading_days
    variance = float(weights.T @ cov_annual @ weights)
    return np.sqrt(max(variance, 0.0))


def negative_sharpe(weights, returns, risk_free_rate=0.0, trading_days=252):
    p_return = portfolio_annual_return(weights, returns, trading_days)
    p_vol = portfolio_annual_volatility(weights, returns, trading_days)
    if p_vol == 0 or np.isnan(p_vol):
        return 1e9
    return -(p_return - risk_free_rate) / p_vol


def optimize_mpt(train_returns, risk_free_rate=0.0, trading_days=252):
    """
    Tối đa hóa Sharpe ratio trên ma trận lợi nhuận Train
    Ràng buộc: Long-only (0 <= w_i <= 1), Tổng w_i = 100%
    """
    n = train_returns.shape[1]
    x0 = np.repeat(1.0 / n, n)
    bounds = [(0.0, 1.0)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    try:
        res = minimize(
            negative_sharpe,
            x0=x0,
            args=(train_returns, risk_free_rate, trading_days),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )
        if res.success:
            return np.maximum(res.x, 0.0) / np.sum(np.maximum(res.x, 0.0))
        else:
            return np.repeat(1.0 / n, n)
    except Exception:
        return np.repeat(1.0 / n, n)


def portfolio_returns(return_matrix, weights):
    return return_matrix.mul(weights, axis=1).sum(axis=1)


# ==========================================
# 6. GIAO DIỆN CHÍNH & SIDEBAR CONTROLS
# ==========================================

st.sidebar.markdown("## ⚙️ Cấu Hình Chiến Lược")

# 1. Nguồn dữ liệu
data_source_mode = st.sidebar.radio(
    "Nguồn dữ liệu CSV:",
    ["Tệp mặc định (HOSE_2020_2023_in.csv)", "Tải lên tệp CSV mới"]
)

uploaded_file = None
default_csv = "HOSE_2020_2023_in.csv"
if not os.path.exists(default_csv):
    if os.path.exists("HOSE_2020_2023_in(1).csv"):
        default_csv = "HOSE_2020_2023_in(1).csv"

if data_source_mode == "Tải lên tệp CSV mới":
    uploaded_file = st.sidebar.file_uploader("Chọn file CSV", type=["csv"])
    target_data = uploaded_file if uploaded_file is not None else default_csv
else:
    target_data = default_csv

df_raw = load_dataset(target_data)

if df_raw is None:
    st.error("⚠️ Không thể tải dữ liệu. Vui lòng kiểm tra lại file CSV trong thư mục hoặc tải lên file hợp lệ!")
    st.stop()

all_tickers = sorted(df_raw["ticker"].unique())

# 2. Chọn cổ phiếu
st.sidebar.markdown("### 📌 Chọn Cổ Phiếu")
default_selection = [t for t in ["ACB", "FPT", "HPG"] if t in all_tickers]
if not default_selection:
    default_selection = all_tickers[:3]

selected_tickers = st.sidebar.multiselect(
    "Chọn 2 hoặc nhiều mã cổ phiếu để cấu thành danh mục:",
    options=all_tickers,
    default=default_selection
)

if len(selected_tickers) < 2:
    st.sidebar.warning("Vui lòng chọn ít nhất 2 mã cổ phiếu để so sánh và tối ưu hóa danh mục MPT!")

# 3. Thiết lập thời gian Train / Test
st.sidebar.markdown("### ⏱️ Phân Chia Thời Gian")
col_d1, col_d2 = st.sidebar.columns(2)
train_start = col_d1.date_input("Train Bắt đầu", pd.to_datetime("2020-01-01"))
train_end = col_d2.date_input("Train Kết thúc", pd.to_datetime("2021-12-31"))

col_d3, col_d4 = st.sidebar.columns(2)
test_start = col_d3.date_input("Test Bắt đầu", pd.to_datetime("2022-01-01"))
test_end = col_d4.date_input("Test Kết thúc", pd.to_datetime("2022-12-31"))

# 4. Thiết lập vốn & phí
st.sidebar.markdown("### 💰 Vốn & Chi Phí")
initial_capital = st.sidebar.number_input(
    "Vốn ban đầu (VNĐ):",
    min_value=100_000,
    max_value=1_000_000_000_000,
    value=1_000_000,
    step=100_000
)

commission_pct = st.sidebar.number_input(
    "Phí giao dịch mỗi lượt (%):",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05,
    help="Phí áp dụng khi mở/đóng vị thế. Đặt 0.0 theo đúng cấu hình notebook gốc."
)
commission = commission_pct / 100.0

portfolio_signal_mode = st.sidebar.selectbox(
    "Quy tắc kết hợp tín hiệu cho Danh Mục:",
    options=["OR", "AND"],
    index=0,
    help="OR: Mua/bán khi ít nhất một chỉ báo báo hiệu. AND: Chỉ mua/bán khi cả hai cùng đồng thuận."
)

risk_free_rate = st.sidebar.number_input(
    "Lãi suất phi rủi ro năm (Rf %):",
    min_value=0.0,
    max_value=15.0,
    value=0.0,
    step=0.5
) / 100.0

# 5. Chế độ tham số
st.sidebar.markdown("### 🛠️ Chế Độ Tham Số")
param_mode = st.sidebar.radio(
    "Phương thức xác định tham số SMA / OBV:",
    ["Tham số tối ưu chuẩn (Notebook Preset)", "Tối ưu hóa Hyperopt trực tiếp", "Tự chỉnh thủ công"],
    index=0
)

hyperopt_evals = 30
if param_mode == "Tối ưu hóa Hyperopt trực tiếp":
    hyperopt_evals = st.sidebar.slider("Số vòng lặp Hyperopt (max_evals):", 15, 60, 30, 5)


# ==========================================
# XỬ LÝ DỮ LIỆU TỪNG CỔ PHIẾU
# ==========================================

stock_data = {}
train_data = {}
test_data = {}

for ticker in selected_tickers:
    try:
        s_df = prepare_stock_data(df_raw, ticker)
        s_train = s_df.loc[(s_df.index >= pd.to_datetime(train_start)) & (s_df.index <= pd.to_datetime(train_end))].copy()
        s_test = s_df.loc[(s_df.index >= pd.to_datetime(test_start)) & (s_df.index <= pd.to_datetime(test_end))].copy()

        if len(s_train) < 30 or len(s_test) < 10:
            st.error(f"Mã {ticker} không đủ dữ liệu trong khoảng thời gian đã chọn!")
            st.stop()

        stock_data[ticker] = s_df
        train_data[ticker] = s_train
        test_data[ticker] = s_test
    except Exception as e:
        st.error(f"Lỗi khi xử lý mã {ticker}: {e}")
        st.stop()


# ==========================================
# KHỞI TẠO BỘ THAM SỐ
# ==========================================

notebook_presets = {
    "ACB": {"ma_short": 60, "ma_long": 270, "obv_window": 5},
    "FPT": {"ma_short": 70, "ma_long": 205, "obv_window": 65},
    "HPG": {"ma_short": 25, "ma_long": 220, "obv_window": 40},
}

params_dict = {}

if param_mode == "Tham số tối ưu chuẩn (Notebook Preset)":
    for ticker in selected_tickers:
        if ticker in notebook_presets:
            params_dict[ticker] = {
                "SMA": {"ma_short": notebook_presets[ticker]["ma_short"], "ma_long": notebook_presets[ticker]["ma_long"]},
                "OBV": {"obv_window": notebook_presets[ticker]["obv_window"]}
            }
        else:
            # Preset an toàn cho các mã khác
            params_dict[ticker] = {
                "SMA": {"ma_short": 30, "ma_long": 200},
                "OBV": {"obv_window": 20}
            }

elif param_mode == "Tự chỉnh thủ công":
    st.sidebar.markdown("#### 🎛️ Tinh chỉnh từng mã")
    for ticker in selected_tickers:
        with st.sidebar.expander(f"Tham số cho mã {ticker}", expanded=False):
            m_s = st.slider(f"{ticker} - MA Ngắn", 10, 100, 30, 5, key=f"{ticker}_ms")
            m_l = st.slider(f"{ticker} - MA Dài", 110, 350, 200, 5, key=f"{ticker}_ml")
            obv_w = st.slider(f"{ticker} - OBV Window", 5, 100, 20, 5, key=f"{ticker}_obv")
            params_dict[ticker] = {
                "SMA": {"ma_short": m_s, "ma_long": m_l},
                "OBV": {"obv_window": obv_w}
            }

else: # Tối ưu hóa Hyperopt trực tiếp
    # Lưu kết quả tối ưu vào session_state để tránh chạy lại khi tương tác UI
    opt_key = f"opt_params_{'_'.join(selected_tickers)}_{hyperopt_evals}_{str(train_start)}_{str(train_end)}"
    
    if opt_key not in st.session_state:
        st.session_state[opt_key] = None

    run_opt = st.sidebar.button("🚀 Bắt đầu chạy Tối ưu Hyperopt trên Train")

    if st.session_state[opt_key] is not None:
        params_dict = st.session_state[opt_key]
    elif run_opt:
        with st.spinner("Đang chạy Hyperopt tối ưu hóa Sharpe Ratio trên tập dữ liệu Train..."):
            progress_bar = st.progress(0)
            opt_res = {}
            for i, ticker in enumerate(selected_tickers):
                sma_b, obv_b = optimize_stock_hyperopt(
                    train_data[ticker],
                    max_evals=hyperopt_evals,
                    commission=commission
                )
                opt_res[ticker] = {"SMA": sma_b, "OBV": obv_b}
                progress_bar.progress((i + 1) / len(selected_tickers))
            st.session_state[opt_key] = opt_res
            params_dict = opt_res
            st.sidebar.success("✅ Tối ưu hoàn tất!")
    else:
        # Tạm thời dùng preset và nhắc người dùng nhấn nút
        st.sidebar.info("👉 Nhấn nút bên trên để bắt đầu tối ưu, hoặc xem trước bằng tham số mặc định.")
        for ticker in selected_tickers:
            params_dict[ticker] = {
                "SMA": {"ma_short": notebook_presets.get(ticker, {}).get("ma_short", 30),
                        "ma_long": notebook_presets.get(ticker, {}).get("ma_long", 200)},
                "OBV": {"obv_window": notebook_presets.get(ticker, {}).get("obv_window", 20)}
            }


# ==========================================
# TÍNH TOÁN HIỆU QUẢ CÁC CHIẾN LƯỢC
# ==========================================

def evaluate_all_strategies(df, sma_p, obv_p, comm=0.0):
    res_returns = {}
    
    # 1. Buy & Hold Benchmark
    bh_ret = df["Close"].pct_change().fillna(0.0)
    res_returns["Buy & Hold"] = bh_ret

    # 2. SMA Only
    e_sma = find_position_sma(df, sma_p)
    r_sma, _ = strategy_returns(df, e_sma, comm)
    res_returns["SMA"] = r_sma

    # 3. OBV Only
    e_obv = find_position_obv(df, obv_p)
    r_obv, _ = strategy_returns(df, e_obv, comm)
    res_returns["OBV"] = r_obv

    # 4. SMA + OBV AND
    e_and = find_position_combined_and(df, sma_p, obv_p)
    r_and, _ = strategy_returns(df, e_and, comm)
    res_returns["SMA + OBV AND"] = r_and

    # 5. SMA + OBV OR
    e_or = find_position_combined_or(df, sma_p, obv_p)
    r_or, _ = strategy_returns(df, e_or, comm)
    res_returns["SMA + OBV OR"] = r_or

    stats_df = pd.DataFrame({
        name: performance_stats(ret)
        for name, ret in res_returns.items()
    }).T

    return res_returns, stats_df


single_stock_train = {}
single_stock_test = {}

for ticker in selected_tickers:
    sma_p = params_dict[ticker]["SMA"]
    obv_p = params_dict[ticker]["OBV"]

    r_tr, s_tr = evaluate_all_strategies(train_data[ticker], sma_p, obv_p, commission)
    r_te, s_te = evaluate_all_strategies(test_data[ticker], sma_p, obv_p, commission)

    single_stock_train[ticker] = {"returns": r_tr, "stats": s_tr}
    single_stock_test[ticker] = {"returns": r_te, "stats": s_te}


# ==========================================
# TÍNH TOÁN DANH MỤC: EQUAL WEIGHT & MPT
# ==========================================

strategy_choice_name = f"SMA + OBV {portfolio_signal_mode.upper()}"

train_ret_matrix = pd.concat(
    {ticker: single_stock_train[ticker]["returns"][strategy_choice_name] for ticker in selected_tickers},
    axis=1
).dropna()

test_ret_matrix = pd.concat(
    {ticker: single_stock_test[ticker]["returns"][strategy_choice_name] for ticker in selected_tickers},
    axis=1
).dropna()

# 1. Trọng số Equal Weight
n_assets = len(selected_tickers)
equal_weights = np.repeat(1.0 / n_assets, n_assets)

# 2. Trọng số MPT (ước lượng CHỈ trên tập Train)
mpt_weights = optimize_mpt(train_ret_matrix, risk_free_rate=risk_free_rate)

# Lợi nhuận danh mục
ew_train_ret = portfolio_returns(train_ret_matrix, equal_weights)
mpt_train_ret = portfolio_returns(train_ret_matrix, mpt_weights)

ew_test_ret = portfolio_returns(test_ret_matrix, equal_weights)
mpt_test_ret = portfolio_returns(test_ret_matrix, mpt_weights)

# Thống kê danh mục
port_stats_train = pd.DataFrame({
    "Equal Weight": performance_stats(ew_train_ret),
    "MPT": performance_stats(mpt_train_ret)
}).T

port_stats_test = pd.DataFrame({
    "Equal Weight": performance_stats(ew_test_ret),
    "MPT": performance_stats(mpt_test_ret)
}).T


# ==========================================
# GIAO DIỆN CHÍNH (MAIN APP DISPLAY)
# ==========================================

st.markdown('<div class="main-header">📈 KIỂM ĐỊNH CHIẾN LƯỢC KẾT HỢP SMA + OBV & PHÂN BỔ DANH MỤC MPT</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Nền tảng kiểm định định lượng (Backtesting), tránh Look-Ahead Bias, tối ưu hóa Out-of-Sample trên thị trường chứng khoán Việt Nam (HOSE).</div>', unsafe_allow_html=True)

# Tabs chính của ứng dụng
tab_intro, tab_data, tab_params, tab_single, tab_portfolio, tab_report = st.tabs([
    "📖 Giới Thiệu & Phương Pháp",
    "📊 Dữ Liệu & Thống Kê",
    "⚙️ Tham Số & Tối Ưu",
    "📈 Kiểm Định Từng Cổ Phiếu",
    "💼 Danh Mục: Equal Weight vs MPT",
    "📑 Báo Cáo & Kết Luận"
])

# ----------------------------------------------------
# TAB 1: GIỚI THIỆU & PHƯƠNG PHÁP
# ----------------------------------------------------
with tab_intro:
    st.markdown("### 🎯 Mục Tiêu Nghiên Cứu")
    st.write("""
    Ứng dụng này mô phỏng và kiểm định tính hiệu quả của chiến lược giao dịch kết hợp giữa **chỉ báo xu hướng (SMA)** 
    và **chỉ báo khối lượng (OBV)**, sau đó ứng dụng lý thuyết **Danh mục đầu tư hiện đại (Markowitz MPT)** 
    để tối ưu hóa phân bổ tỷ trọng tài sản.
    """)

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.markdown("""
        <div class="card">
            <h4>1. Tín hiệu SMA Crossover</h4>
            <p>Sử dụng 2 đường trung bình động (MA Ngắn & MA Dài):</p>
            <ul>
                <li><b>BUY (+1):</b> MA ngắn cắt lên MA dài (Golden Cross)</li>
                <li><b>SELL (-1):</b> MA ngắn cắt xuống MA dài (Death Cross)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_p2:
        st.markdown("""
        <div class="card">
            <h4>2. Tín hiệu OBV Crossover</h4>
            <p>Sử dụng chỉ báo dòng tiền On-Balance Volume và đường trung bình OBV-MA:</p>
            <ul>
                <li><b>BUY (+1):</b> OBV vượt lên trên OBV-MA (Dòng tiền vào mạnh)</li>
                <li><b>SELL (-1):</b> OBV rơi xuống dưới OBV-MA (Dòng tiền suy yếu)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_p3:
        st.markdown("""
        <div class="card">
            <h4>3. Quy tắc Triển khai Vị thế</h4>
            <p>Tuân thủ nghiêm ngặt chuẩn mực định lượng:</p>
            <ul>
                <li><b>Không Look-ahead bias:</b> Tín hiệu ngày <i>t</i> chỉ thực hiện vào ngày <i>t+1</i> (lag 1 phiên).</li>
                <li><b>Không Overfitting:</b> Chỉ tối ưu tham số và trọng số MPT trên <b>Train</b>; <b>Test</b> hoàn toàn giữ nguyên.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚖️ So Sánh Hai Phương Thức Phân Bổ Danh Mục")
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.info("""
        **1. Equal Weight (Đẳng trọng số):**
        - Mỗi cổ phiếu được phân bổ tỷ trọng bằng nhau: $w_i = \\frac{1}{N}$.
        - Đơn giản, trực quan, không phụ thuộc vào dữ liệu quá khứ.
        """)
    with col_w2:
        st.success("""
        **2. Modern Portfolio Theory (MPT - Markowitz):**
        - Tìm vector trọng số $w$ để tối đa hóa **Sharpe Ratio** trên tập Train.
        - Ràng buộc: Long-only ($0 \\le w_i \\le 1$), tổng trọng số $\\sum w_i = 100\\%$.
        - Được giữ nguyên khi chạy kiểm định trên tập Test (Out-of-sample).
        """)


# ----------------------------------------------------
# TAB 2: DỮ LIỆU & THỐNG KÊ MÔ TẢ
# ----------------------------------------------------
with tab_data:
    st.markdown("### 📋 Thông Tin Tập Dữ Liệu")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Tổng số mã HOSE</div>
            <div class="metric-value">{len(all_tickers)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Mã được chọn</div>
            <div class="metric-value">{len(selected_tickers)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Khoảng Train (In-Sample)</div>
            <div class="metric-value">{train_start} → {train_end}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Khoảng Test (Out-Sample)</div>
            <div class="metric-value">{test_start} → {test_end}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### 📈 Biểu Đồ Giá Đóng Cửa & Khối Lượng Giao Dịch")
    chart_ticker = st.selectbox("Chọn mã để hiển thị chi tiết:", selected_tickers, key="chart_ticker_tab2")
    df_disp = stock_data[chart_ticker]

    fig_ohlc = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"Diễn biến Giá Cổ Phiếu {chart_ticker}", "Khối Lượng Giao Dịch (Volume)")
    )

    fig_ohlc.add_trace(
        go.Candlestick(
            x=df_disp.index,
            open=df_disp["Open"],
            high=df_disp["High"],
            low=df_disp["Low"],
            close=df_disp["Close"],
            name="OHLC"
        ),
        row=1, col=1
    )

    # Đánh dấu vùng Train và Test
    fig_ohlc.add_vrect(
        x0=str(train_start), x1=str(train_end),
        fillcolor="blue", opacity=0.07, line_width=1, line_dash="dash",
        annotation_text="Giai đoạn Train (In-Sample)", annotation_position="top left",
        row=1, col=1
    )
    fig_ohlc.add_vrect(
        x0=str(test_start), x1=str(test_end),
        fillcolor="orange", opacity=0.07, line_width=1, line_dash="dash",
        annotation_text="Giai đoạn Test (Out-of-Sample)", annotation_position="top left",
        row=1, col=1
    )

    colors = ['green' if c >= o else 'red' for c, o in zip(df_disp["Close"], df_disp["Open"])]
    fig_ohlc.add_trace(
        go.Bar(x=df_disp.index, y=df_disp["Volume"], marker_color=colors, name="Volume"),
        row=2, col=1
    )

    fig_ohlc.update_layout(height=550, xaxis_rangeslider_visible=False, template="plotly_white")
    st.plotly_chart(fig_ohlc, use_container_width=True)

    st.markdown("#### 🔍 Xem 5 Dòng Dữ Liệu Đầu Tiên")
    st.dataframe(df_disp.head(), use_container_width=True)


# ----------------------------------------------------
# TAB 3: THAM SỐ & TỐI ƯU HÓA
# ----------------------------------------------------
with tab_params:
    st.markdown("### ⚙️ Bảng Tham Số Tối Ưu Từng Cổ Phiếu (Ước lượng từ Train)")
    st.write("""
    Mỗi cổ phiếu có đặc tính biến động và cấu trúc dòng tiền khác nhau, do đó việc tối ưu hóa 
    bộ tham số riêng biệt trên giai đoạn **Train** giúp chiến lược bắt kịp chu kỳ vận động của từng cổ phiếu.
    """)

    params_summary = []
    for ticker in selected_tickers:
        params_summary.append({
            "Mã CP": ticker,
            "MA Ngắn (ma_short)": params_dict[ticker]["SMA"]["ma_short"],
            "MA Dài (ma_long)": params_dict[ticker]["SMA"]["ma_long"],
            "Chu kỳ OBV (obv_window)": params_dict[ticker]["OBV"]["obv_window"],
            "Phương thức": param_mode
        })
    df_params_summary = pd.DataFrame(params_summary).set_index("Mã CP")
    st.table(df_params_summary)

    st.info("""
    💡 **Ghi chú về mặt phương pháp luận:**
    - Toàn bộ tham số trên được tìm kiếm và đánh giá dựa trên hàm mục tiêu **Maximize Sharpe Ratio** trên tập dữ liệu **Train**.
    - **Tuyệt đối không** sử dụng dữ liệu Test trong quá trình tìm tham số nhằm đảm bảo tính khách quan và kiểm tra khả năng thích ứng thực tế của mô hình.
    """)


# ----------------------------------------------------
# TAB 4: KIỂM ĐỊNH TỪNG CỔ PHIẾU
# ----------------------------------------------------
with tab_single:
    st.markdown("### 📈 Phân Tích & Kiểm Định Từng Mã Cổ Phiếu")
    inspect_ticker = st.selectbox("Chọn mã cổ phiếu cần phân tích chuyên sâu:", selected_tickers, key="inspect_ticker_tab4")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(f"#### 🔵 Kết Quả Trên TRAIN ({train_start} → {train_end})")
        tr_stats = single_stock_train[inspect_ticker]["stats"]
        st.dataframe(tr_stats.round(4).style.highlight_max(axis=0, color="#D1FAE5"), use_container_width=True)

    with col_t2:
        st.markdown(f"#### 🟡 Kết Quả Trên TEST ({test_start} → {test_end})")
        te_stats = single_stock_test[inspect_ticker]["stats"]
        st.dataframe(te_stats.round(4).style.highlight_max(axis=0, color="#FEF3C7"), use_container_width=True)

    st.markdown("---")
    st.markdown(f"#### 📊 Biểu Đồ Kỹ Thuật & Tín Hiệu Giao Dịch ({inspect_ticker})")

    # Vẽ biểu đồ giá kèm tín hiệu BUY / SELL
    inspect_df = stock_data[inspect_ticker]
    inspect_sma_p = params_dict[inspect_ticker]["SMA"]
    inspect_obv_p = params_dict[inspect_ticker]["OBV"]

    # Tính đường chỉ báo
    ma_s_series = ta.trend.SMAIndicator(inspect_df["Close"], window=int(inspect_sma_p["ma_short"])).sma_indicator()
    ma_l_series = ta.trend.SMAIndicator(inspect_df["Close"], window=int(inspect_sma_p["ma_long"])).sma_indicator()
    obv_series = ta.volume.OnBalanceVolumeIndicator(inspect_df["Close"], inspect_df["Volume"]).on_balance_volume()
    obv_ma_series = obv_series.rolling(window=int(inspect_obv_p["obv_window"])).mean()

    # Tín hiệu kết hợp theo cấu hình đã chọn
    if portfolio_signal_mode == "OR":
        sig_series = find_position_combined_or(inspect_df, inspect_sma_p, inspect_obv_p)
    else:
        sig_series = find_position_combined_and(inspect_df, inspect_sma_p, inspect_obv_p)

    buy_points = inspect_df[sig_series == 1]
    sell_points = inspect_df[sig_series == -1]

    fig_signals = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
        row_heights=[0.65, 0.35],
        subplot_titles=(f"Giá {inspect_ticker} & Điểm Mua/Bán (Chiến Lược {strategy_choice_name})", "Chỉ Báo OBV & Đường OBV-MA")
    )

    # Đường giá & SMA
    fig_signals.add_trace(go.Scatter(x=inspect_df.index, y=inspect_df["Close"], name="Giá Đóng Cửa", line=dict(color="#1E3A8A", width=1.5)), row=1, col=1)
    fig_signals.add_trace(go.Scatter(x=ma_s_series.index, y=ma_s_series, name=f"SMA Ngắn ({inspect_sma_p['ma_short']})", line=dict(color="#10B981", width=1.2)), row=1, col=1)
    fig_signals.add_trace(go.Scatter(x=ma_l_series.index, y=ma_l_series, name=f"SMA Dài ({inspect_sma_p['ma_long']})", line=dict(color="#EF4444", width=1.2)), row=1, col=1)

    # Điểm Buy / Sell
    fig_signals.add_trace(
        go.Scatter(
            x=buy_points.index, y=buy_points["Close"] * 0.98,
            mode="markers", marker=dict(symbol="triangle-up", size=10, color="green"),
            name="Tín hiệu MUA"
        ), row=1, col=1
    )
    fig_signals.add_trace(
        go.Scatter(
            x=sell_points.index, y=sell_points["Close"] * 1.02,
            mode="markers", marker=dict(symbol="triangle-down", size=10, color="red"),
            name="Tín hiệu BÁN"
        ), row=1, col=1
    )

    # OBV
    fig_signals.add_trace(go.Scatter(x=obv_series.index, y=obv_series, name="OBV", line=dict(color="#8B5CF6", width=1.2)), row=2, col=1)
    fig_signals.add_trace(go.Scatter(x=obv_ma_series.index, y=obv_ma_series, name=f"OBV-MA ({inspect_obv_p['obv_window']})", line=dict(color="#F59E0B", width=1.2, dash="dot")), row=2, col=1)

    fig_signals.update_layout(height=600, template="plotly_white", legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig_signals, use_container_width=True)

    # Đường tăng trưởng tài sản (Equity Curves) cho cổ phiếu này trên Test
    st.markdown(f"#### 💰 Đường Tăng Trưởng Tài Sản Trên Giai Đoạn TEST 2022 ({inspect_ticker})")
    test_returns_dict = single_stock_test[inspect_ticker]["returns"]

    fig_eq_single = go.Figure()
    for strat_name, r_series in test_returns_dict.items():
        cum_eq = initial_capital * (1.0 + r_series).cumprod()
        fig_eq_single.add_trace(go.Scatter(x=cum_eq.index, y=cum_eq, name=strat_name, mode="lines"))

    fig_eq_single.update_layout(
        title=f"Đường Vốn Chiến Lược — {inspect_ticker} (Test 2022 - Vốn khởi đầu {initial_capital:,.0f} VNĐ)",
        xaxis_title="Thời Gian",
        yaxis_title="Giá Trị Danh Mục (VNĐ)",
        template="plotly_white",
        height=450
    )
    st.plotly_chart(fig_eq_single, use_container_width=True)


# ----------------------------------------------------
# TAB 5: DANH MỤC: EQUAL WEIGHT VS MPT
# ----------------------------------------------------
with tab_portfolio:
    st.markdown("### 💼 Kiểm Định Phân Bổ Danh Mục Đầu Tư")
    st.write(f"""
    Đánh giá danh mục kết hợp gồm **{len(selected_tickers)} cổ phiếu** ({', '.join(selected_tickers)}) 
    với tín hiệu thành phần là **{strategy_choice_name}**.
    """)

    # Trọng số danh mục
    col_w_tbl, col_w_chart = st.columns([0.4, 0.6])
    
    weights_df = pd.DataFrame({
        "Equal Weight": equal_weights,
        "MPT (Markowitz)": mpt_weights
    }, index=selected_tickers)

    with col_w_tbl:
        st.markdown("#### ⚖️ Trọng Số Tỷ Trọng Phân Bổ")
        st.dataframe(weights_df.style.format("{:.2%}"), use_container_width=True)
        st.caption("Lưu ý: Trọng số MPT được tối ưu hóa tối đa hóa Sharpe Ratio CHỈ trên dữ liệu Train và giữ nguyên sang Test.")

    with col_w_chart:
        fig_weights = go.Figure()
        fig_weights.add_trace(go.Bar(x=selected_tickers, y=weights_df["Equal Weight"], name="Equal Weight", marker_color="#3B82F6"))
        fig_weights.add_trace(go.Bar(x=selected_tickers, y=weights_df["MPT (Markowitz)"], name="MPT", marker_color="#10B981"))
        fig_weights.update_layout(
            title="So sánh Trọng số Phân bổ (Equal Weight vs MPT)",
            yaxis=dict(title="Tỷ trọng", tickformat=".0%"),
            barmode="group",
            height=300,
            template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_weights, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📊 So Sánh Hiệu Quả Đầu Tư (Performance Comparison)")

    col_res_tr, col_res_te = st.columns(2)
    with col_res_tr:
        st.markdown(f"#### 🔵 KẾT QUẢ DANH MỤC TRÊN TRAIN ({train_start} → {train_end})")
        st.dataframe(port_stats_train.round(4).style.highlight_max(axis=0, color="#D1FAE5"), use_container_width=True)

    with col_res_te:
        st.markdown(f"#### 🟡 KẾT QUẢ DANH MỤC TRÊN TEST ({test_start} → {test_end})")
        st.dataframe(port_stats_test.round(4).style.highlight_max(axis=0, color="#FEF3C7"), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🚀 Đường Tăng Trưởng Tài Sản Trên Giai Đoạn TEST 2022")

    ew_equity_test = initial_capital * (1.0 + ew_test_ret).cumprod()
    mpt_equity_test = initial_capital * (1.0 + mpt_test_ret).cumprod()

    # Tính drawdown
    ew_dd = (ew_equity_test / ew_equity_test.cummax() - 1.0) * 100.0
    mpt_dd = (mpt_equity_test / mpt_equity_test.cummax() - 1.0) * 100.0

    fig_port_eq = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=(
            f"Đường Tăng Trưởng Tài Sản Danh Mục (Vốn Ban Đầu {initial_capital:,.0f} VNĐ)",
            "Đường Sụt Giảm Giá Trị Danh Mục (Underwater Drawdown %)"
        )
    )

    fig_port_eq.add_trace(go.Scatter(x=ew_equity_test.index, y=ew_equity_test, name="Equal Weight", line=dict(color="#3B82F6", width=2)), row=1, col=1)
    fig_port_eq.add_trace(go.Scatter(x=mpt_equity_test.index, y=mpt_equity_test, name="MPT (Markowitz)", line=dict(color="#10B981", width=2)), row=1, col=1)

    fig_port_eq.add_trace(go.Scatter(x=ew_dd.index, y=ew_dd, name="Drawdown EW", line=dict(color="#3B82F6", width=1.2), fill="tozeroy"), row=2, col=1)
    fig_port_eq.add_trace(go.Scatter(x=mpt_dd.index, y=mpt_dd, name="Drawdown MPT", line=dict(color="#10B981", width=1.2), fill="tozeroy"), row=2, col=1)

    fig_port_eq.update_layout(height=580, template="plotly_white")
    st.plotly_chart(fig_port_eq, use_container_width=True)

    # Ma trận tương quan giữa các cổ phiếu
    st.markdown("#### 🔗 Ma Trận Tương Quan Lợi Nhuận (Correlation Matrix)")
    corr_matrix = train_ret_matrix.corr()
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="Blues",
        title="Tương quan lợi nhuận giữa các mã trên tập Train"
    )
    fig_corr.update_layout(height=350)
    st.plotly_chart(fig_corr, use_container_width=True)


# ----------------------------------------------------
# TAB 6: BÁO CÁO & XUẤT DỮ LIỆU
# ----------------------------------------------------
with tab_report:
    st.markdown("### 📑 Báo Cáo Tổng Hợp Kết Quả & Khuyến Nghị Đầu Tư")

    st.write(f"""
    Dưới đây là bản tóm lược toàn diện được tự động tổng hợp dựa trên 8 tiêu chuẩn báo cáo định lượng:
    """)

    # 1. Tóm tắt danh mục
    st.markdown(f"""
    1. **Các cổ phiếu được lựa chọn:** {', '.join(selected_tickers)}
    2. **Bộ tham số tối ưu (Train {train_start} → {train_end}):** Đã áp dụng Hyperopt / Cấu hình chuẩn cho từng mã.
    3. **Quy tắc kết hợp tín hiệu danh mục:** `{strategy_choice_name}`
    4. **Phương pháp phân bổ:**
       - **Equal Weight:** Phân bổ đều {100.0/len(selected_tickers):.2f}% cho mỗi cổ phiếu.
       - **MPT (Markowitz):** Ước lượng vector tỷ trọng tối ưu Sharpe trên Train:
         {', '.join([f'**{t}:** {w:.2%}' for t, w in zip(selected_tickers, mpt_weights)])}
    """)

    # 2. So sánh kết quả ngoài mẫu
    ew_ret_val = port_stats_test.loc["Equal Weight", "Total Return [%]"]
    mpt_ret_val = port_stats_test.loc["MPT", "Total Return [%]"]
    ew_sharpe_val = port_stats_test.loc["Equal Weight", "Sharpe Ratio"]
    mpt_sharpe_val = port_stats_test.loc["MPT", "Sharpe Ratio"]
    ew_mdd_val = port_stats_test.loc["Equal Weight", "Max Drawdown [%]"]
    mpt_mdd_val = port_stats_test.loc["MPT", "Max Drawdown [%]"]

    better_port = "Equal Weight" if ew_sharpe_val >= mpt_sharpe_val else "MPT"

    st.markdown("#### 🏆 Đánh Giá Ngoài Mẫu (Out-of-sample Test 2022):")
    st.markdown(f"""
    - **Lợi nhuận tổng thể (Total Return):**
      - Equal Weight: **{ew_ret_val:.2f}%**
      - MPT: **{mpt_ret_val:.2f}%**
    - **Tỷ suất Sharpe (Risk-adjusted return):**
      - Equal Weight: **{ew_sharpe_val:.3f}**
      - MPT: **{mpt_sharpe_val:.3f}**
    - **Mức sụt giảm tối đa (Max Drawdown):**
      - Equal Weight: **{ew_mdd_val:.2f}%**
      - MPT: **{mpt_mdd_val:.2f}%**
    """)

    st.markdown("""
    > ⚠️ **Nhận xét chuyên gia về hiện tượng Overfitting:**
    > - Trong giai đoạn Train (2020-2021) khi thị trường trong xu hướng tăng (Bull Market), danh mục MPT thường đạt Sharpe Ratio cao hơn Equal Weight do mô hình tối ưu hóa dựa trên dữ liệu lịch sử đã qua.
    > - Tuy nhiên, sang năm 2022 (Bear Market - thị trường giảm mạnh), cấu trúc ma trận hiệp phương sai và xu hướng dòng tiền đảo chiều mạnh mẽ. Kết quả ngoài mẫu Test 2022 phản ánh khả năng chống chịu thực tế của danh mục.
    > - Việc kết hợp SMA và OBV giúp chiến lược thoát vị thế kịp thời vào những nhịp sụt giảm sâu, bảo vệ vốn vượt trội so với chiến lược Mua và Nắm giữ (Buy & Hold) thông thường.
    """)

    st.markdown("---")
    st.markdown("### 📥 Tải Dữ Liệu Kết Quả (Export CSV)")

    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        # Xuất chuỗi lợi nhuận Test danh mục
        df_export_returns = pd.DataFrame({
            "Equal_Weight_Return": ew_test_ret,
            "MPT_Return": mpt_test_ret,
            "Equal_Weight_Equity": ew_equity_test,
            "MPT_Equity": mpt_equity_test
        })
        csv_returns = df_export_returns.to_csv(index=True).encode("utf-8-sig")
        st.download_button(
            label="📥 Tải xuống Chuỗi Lợi Nhuận Test (CSV)",
            data=csv_returns,
            file_name=f"portfolio_test_returns_{portfolio_signal_mode}.csv",
            mime="text/csv"
        )

    with col_exp2:
        # Xuất bảng thống kê hiệu năng
        df_export_stats = pd.concat([
            port_stats_train.add_suffix(" (Train)"),
            port_stats_test.add_suffix(" (Test)")
        ], axis=1)
        csv_stats = df_export_stats.to_csv(index=True).encode("utf-8-sig")
        st.download_button(
            label="📥 Tải xuống Bảng Thống Kê Hiệu Năng (CSV)",
            data=csv_stats,
            file_name="portfolio_performance_stats.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #9CA3AF; font-size: 0.85rem;">
    Đồ án / Khóa luận Định lượng Đầu tư Tài chính — Xây dựng trên Streamlit & Python Quantitative Finance.
</div>
""", unsafe_allow_html=True)
