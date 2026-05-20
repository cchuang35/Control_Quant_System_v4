# 以控制觀點設計量化交易系統：五層架構 v1

## 0. 系統總覽

本文件整理目前定案的 **五層控制式量化交易系統 v1**。  
此版本的目標不是直接設計一個「保證賺錢」的交易策略，而是建立一個可解釋、可回測、可逐層升級的控制系統骨架。

五層架構如下：

```text
Layer 1：市場建模 Market Modeling
Layer 2：狀態估測 State Estimation
Layer 3：策略控制器 Strategy Controller
Layer 4：風險與穩定性分析 Risk & Stability Analysis
Layer 5：自適應與學習 Adaptation & Learning
```

整體資料流：

```text
OHLCV
  ↓
Layer 1：MarketStateV1
  ↓
Layer 2：EstimatedMarketStateV1
  ↓
Layer 3：ControlActionV1
  ↓
Layer 4：SafeControlActionV1
  ↓
Layer 5：AdaptiveUpdateV1
```

核心精神：

```text
先感知市場 → 再估測狀態 → 再決定曝險 → 再檢查風險 → 最後根據回饋調整系統
```

---

# 1. Layer 1：市場建模 Market Modeling v1

## 1.1 定位

Layer 1 的任務是：

> 將原始市場資料轉換成 controller 可以使用的市場狀態表示。

它不是交易策略，也不是買賣訊號。  
它比較像是整個系統的「感測器與前處理層」。

---

## 1.2 Input

v1 只使用 OHLCV：

```text
open
high
low
close
volume
```

也就是說，只要有 K 線資料，就可以先跑 v1。

---

## 1.3 Output

Layer 1 輸出：

```text
MarketStateV1
```

定義：

```text
MarketStateV1_t = {
    timestamp,
    close,

    return_1,

    volatility,
    volatility_score,

    trend_raw,
    trend_score,

    volume_z,
    volume_score,

    price_range,
    liquidity_score,

    drawdown,

    shock_score,

    confidence,

    market_mode
}
```

核心狀態簡化版：

```text
MarketStateV1_t = {
    trend_score_t,
    volatility_score_t,
    volume_score_t,
    liquidity_score_t,
    drawdown_t,
    shock_score_t,
    confidence_t,
    market_mode_t
}
```

---

## 1.4 核心變數

### 1.4.1 Return

使用 log return：

```text
r_t = log(Close_t / Close_{t-1})
```

用途：

```text
計算波動率
計算 shock
輔助後續 regime estimation
```

---

### 1.4.2 Volatility Score

使用 rolling volatility：

```text
vol_t = std(r_{t-19}, ..., r_t)
```

使用過去一段時間的 median 作為 baseline：

```text
volatility_score_t = vol_t / rolling_median(vol_t, 120)
```

解讀：

| volatility_score | 狀態 |
|---|---|
| `< 0.8` | low volatility |
| `0.8 ~ 1.5` | normal volatility |
| `1.5 ~ 2.5` | high volatility |
| `> 2.5` | extreme volatility |

---

### 1.4.3 Trend Score

使用短長均線差：

```text
MA_short = moving_average(close, 20)
MA_long  = moving_average(close, 60)
```

```text
trend_raw_t = (MA_short_t - MA_long_t) / MA_long_t
```

壓縮到 `[-1, 1]`：

```text
trend_score_t = tanh(k * trend_raw_t)
```

解讀：

| trend_score | 意義 |
|---|---|
| 接近 `+1` | 強上升趨勢 |
| 接近 `0` | 沒明顯趨勢 |
| 接近 `-1` | 強下降趨勢 |

---

### 1.4.4 Volume Score

使用成交量 z-score：

```text
volume_z_t = (volume_t - mean(volume, 60)) / std(volume, 60)
```

轉成異常分數：

```text
volume_score_t = clip(abs(volume_z_t) / 3, 0, 1)
```

解讀：

| volume_score | 意義 |
|---|---|
| `0 ~ 0.3` | 成交量正常 |
| `0.3 ~ 0.7` | 成交量偏異常 |
| `0.7 ~ 1.0` | 成交量非常異常 |

---

### 1.4.5 Liquidity Score

v1 沒有 order book，所以用 proxy：

```text
price_range_t = (High_t - Low_t) / Close_t
```

```text
normalized_volume_t = volume_t / mean(volume, 60)
```

```text
illiquidity_t = price_range_t / normalized_volume_t
```

再正規化成：

```text
liquidity_score_t = 1 - normalized_illiquidity_t
```

解讀：

| liquidity_score | 意義 |
|---|---|
| 接近 `1` | 流動性較好 |
| 接近 `0` | 流動性較差 |

---

### 1.4.6 Drawdown

使用近期高點計算：

```text
rolling_high_t = max(Close_{t-119}, ..., Close_t)
drawdown_t = Close_t / rolling_high_t - 1
```

例如：

```text
drawdown_t = -0.15
```

代表目前價格距離近期高點下跌 15%。

---

### 1.4.7 Shock Score

使用 return z-score：

```text
return_z_t = abs(r_t) / rolling_std(r_t, 60)
```

```text
shock_score_t = clip(return_z_t / 4, 0, 1)
```

解讀：

| shock_score | 意義 |
|---|---|
| `0 ~ 0.3` | 正常 |
| `0.3 ~ 0.7` | 有異常波動 |
| `0.7 ~ 1.0` | 可能是突發事件 |

---

### 1.4.8 Confidence

v1 使用 rule-based confidence：

```text
confidence_t = 1
               - 0.3 * shock_score_t
               - 0.3 * volatility_penalty_t
               - 0.2 * illiquidity_penalty_t
               - 0.2 * missing_data_penalty_t
```

其中：

```text
volatility_penalty_t = clip((volatility_score_t - 1.5) / 2, 0, 1)
illiquidity_penalty_t = 1 - liquidity_score_t
```

---

## 1.5 Market Mode

v1 使用 rule-based 分類：

```text
market_mode_t ∈ {
    normal,
    trending_up,
    trending_down,
    high_volatility,
    stressed,
    shock
}
```

規則：

```text
if shock_score > 0.8:
    market_mode = "shock"

else if volatility_score > 2.5 and liquidity_score < 0.4:
    market_mode = "stressed"

else if volatility_score > 1.8:
    market_mode = "high_volatility"

else if trend_score > 0.4:
    market_mode = "trending_up"

else if trend_score < -0.4:
    market_mode = "trending_down"

else:
    market_mode = "normal"
```

---

## 1.6 C++ 資料結構

```cpp
struct MarketStateV1 {
    double timestamp;
    double close;

    double return_1;

    double volatility;
    double volatility_score;

    double trend_raw;
    double trend_score;

    double volume_z;
    double volume_score;

    double price_range;
    double liquidity_score;

    double drawdown;

    double shock_score;

    double confidence;

    std::string market_mode;
};
```

---

## 1.7 Layer 1 v1 目標

Layer 1 的目標不是預測價格，而是：

```text
把 OHLCV 轉換成：
- 趨勢狀態
- 波動狀態
- 成交量異常
- 流動性 proxy
- 回撤狀態
- shock 狀態
- confidence
```

一句話：

> Layer 1 是交易控制系統的市場感測器。

---

# 2. Layer 2：狀態估測 State Estimation v1

## 2.1 定位

Layer 2 的任務是：

> 根據 Layer 1 輸出的 MarketStateV1，估計不可觀測的市場 regime 與風險信念。

Layer 2 不直接下單，而是回答：

```text
現在市場比較像哪一種 regime？
這個判斷有多確定？
市場是否正在切換狀態？
目前危險程度多高？
```

---

## 2.2 Input

來自 Layer 1：

```text
z_t = [
    trend_score_t,
    volatility_score_t,
    volume_score_t,
    liquidity_score_t,
    drawdown_t,
    shock_score_t,
    confidence_t
]
```

`market_mode_t` 可以保留作為 debug 與人類閱讀，但 v1 不建議當作主要輸入。

---

## 2.3 Regime Set

v1 使用五種 regime：

```text
R = {
    bull,
    bear,
    sideways,
    high_vol,
    crash_risk
}
```

| Regime | 意義 |
|---|---|
| `bull` | 趨勢偏多、風險正常 |
| `bear` | 趨勢偏空、風險正常或偏高 |
| `sideways` | 沒明顯方向、震盪 |
| `high_vol` | 波動顯著升高，但不一定崩盤 |
| `crash_risk` | 高波動、低流動性、shock、嚴重回撤 |

---

## 2.4 Output

Layer 2 輸出：

```text
EstimatedMarketStateV1
```

定義：

```text
EstimatedMarketStateV1_t = {
    p_bull,
    p_bear,
    p_sideways,
    p_high_vol,
    p_crash_risk,

    dominant_regime,

    state_confidence,
    regime_uncertainty,
    transition_risk,
    danger_score
}
```

---

## 2.5 輔助變數

```text
up_trend      = max(trend_score, 0)
down_trend    = max(-trend_score, 0)
flatness      = 1 - abs(trend_score)

high_vol      = clip((volatility_score - 1.2) / 1.8, 0, 1)
extreme_vol   = clip((volatility_score - 2.0) / 1.5, 0, 1)

illiquidity   = 1 - liquidity_score
drawdown_sev  = clip(abs(drawdown) / 0.25, 0, 1)
shock         = shock_score
```

---

## 2.6 Regime Score

### Bull Score

```text
bull_score =
    1.2 * up_trend
  + 0.4 * confidence
  + 0.2 * liquidity_score
  - 0.5 * high_vol
  - 0.7 * shock
  - 0.5 * drawdown_sev
```

---

### Bear Score

```text
bear_score =
    1.2 * down_trend
  + 0.4 * drawdown_sev
  + 0.3 * high_vol
  - 0.4 * shock
  - 0.2 * illiquidity
```

---

### Sideways Score

```text
sideways_score =
    1.2 * flatness
  + 0.3 * liquidity_score
  + 0.2 * confidence
  - 0.6 * high_vol
  - 0.8 * shock
```

---

### High Vol Score

```text
high_vol_score =
    1.2 * high_vol
  + 0.4 * volume_score
  + 0.3 * abs(trend_score)
  - 0.4 * shock
```

---

### Crash Risk Score

```text
crash_risk_score =
    1.2 * shock
  + 1.0 * extreme_vol
  + 0.8 * illiquidity
  + 0.8 * drawdown_sev
  + 0.4 * volume_score
  - 0.3 * confidence
```

---

## 2.7 Softmax Probability

將 raw score 轉成 regime probability：

```text
P_i = exp(score_i / T) / sum_j exp(score_j / T)
```

v1 預設：

```text
T = 0.7
```

---

## 2.8 Temporal Smoothing

避免 regime probability 每根 K 線劇烈跳動：

```text
P_t = α * P_raw_t + (1 - α) * P_{t-1}
```

v1 預設：

```text
α = 0.2
```

---

## 2.9 State Confidence

```text
regime_clarity = max(P_bull, P_bear, P_sideways, P_high_vol, P_crash_risk)
```

```text
state_confidence = phase1_confidence * regime_clarity
```

---

## 2.10 Regime Uncertainty

使用 normalized entropy：

```text
H(P) = - sum_i P_i log(P_i)
```

```text
regime_uncertainty = H(P) / log(N)
```

其中：

```text
N = 5
```

---

## 2.11 Transition Risk

```text
probability_shift = 0.5 * sum_i |P_t(i) - P_{t-1}(i)|
```

```text
transition_risk = 0.7 * probability_shift + 0.3 * shock_score
```

---

## 2.12 Danger Score

```text
danger_score =
    0.35 * P(crash_risk)
  + 0.25 * P(high_vol)
  + 0.20 * shock_score
  + 0.10 * (1 - liquidity_score)
  + 0.10 * drawdown_severity
```

解讀：

| danger_score | 狀態 |
|---|---|
| `0 ~ 0.3` | 正常 |
| `0.3 ~ 0.6` | 需要保守 |
| `0.6 ~ 0.8` | 危險 |
| `> 0.8` | 極危險 |

---

## 2.13 C++ 資料結構

```cpp
struct EstimatedMarketStateV1 {
    double p_bull;
    double p_bear;
    double p_sideways;
    double p_high_vol;
    double p_crash_risk;

    std::string dominant_regime;

    double state_confidence;
    double regime_uncertainty;
    double transition_risk;
    double danger_score;
};
```

---

## 2.14 Layer 2 v1 目標

Layer 2 的目標是：

```text
1. 估計市場目前最可能的 regime
2. 量化 regime 的不確定性
3. 偵測 regime 是否正在切換
4. 輸出 danger_score 給後續風控
5. 給 Layer 3 一組穩定的 regime belief
```

一句話：

> Layer 2 是把市場狀態轉成 regime belief 與風險信念。

---

# 3. Layer 3：策略控制器 Strategy Controller v1

## 3.1 定位

Layer 3 的任務是：

> 根據市場狀態與估測狀態，決定應該承擔多少交易曝險。

Layer 3 不直接輸出實際下單，而是輸出 desired control action。

---

## 3.2 v1 控制器類型

v1 採用：

```text
Rule-based Risk-aware Exposure Controller
```

中文：

```text
規則式風險感知曝險控制器
```

---

## 3.3 Input

Layer 3 使用三類 input：

```text
MarketStateV1
EstimatedMarketStateV1
PortfolioStateV1
```

---

## 3.4 PortfolioStateV1

```text
PortfolioStateV1 = {
    current_exposure,
    current_position,
    equity,
    cash,
    unrealized_pnl,
    portfolio_drawdown,
    leverage,
    available_margin
}
```

v1 最重要的是：

```text
current_exposure
portfolio_drawdown
leverage
```

---

## 3.5 Output

Layer 3 輸出：

```text
ControlActionV1
```

定義：

```text
ControlActionV1 = {
    target_exposure,
    exposure_change,

    max_leverage,
    rebalance_speed,

    trade_allowed,
    reduce_only,

    action_type,
    reason_code
}
```

---

## 3.6 Target Exposure

核心輸出是：

```text
target_exposure ∈ [-1, 1]
```

| target_exposure | 意義 |
|---|---|
| `+1` | 滿多 |
| `+0.5` | 半多 |
| `0` | 空手 |
| `-0.5` | 半空 |
| `-1` | 滿空 |

若只做現貨多單，則限制為：

```text
target_exposure ∈ [0, 1]
```

---

## 3.7 Step 1：Directional Signal

```text
directional_signal =
    0.6 * (p_bull - p_bear)
  + 0.4 * trend_score
```

解讀：

```text
directional_signal > 0：偏多
directional_signal < 0：偏空
directional_signal ≈ 0：中性
```

最後限制在：

```text
[-1, 1]
```

---

## 3.8 Step 2：Risk Scaler

定義：

```text
risk_scaler ∈ [0, 1]
```

子項：

```text
risk_from_danger = 1 - danger_score
risk_from_vol = 1 - 0.5 * p_high_vol
risk_from_crash = 1 - 0.8 * p_crash_risk
risk_from_confidence = state_confidence
risk_from_transition = 1 - 0.6 * transition_risk
```

總風險縮放：

```text
risk_scaler =
    risk_from_danger
  * risk_from_vol
  * risk_from_crash
  * risk_from_confidence
  * risk_from_transition
```

最後：

```text
risk_scaler = clip(risk_scaler, 0, 1)
```

---

## 3.9 Step 3：Sideways Scaler

```text
sideways_scaler = 1 - 0.5 * p_sideways
```

---

## 3.10 Step 4：Target Exposure

```text
target_exposure =
    directional_signal
  * risk_scaler
  * sideways_scaler
```

---

## 3.11 Trade Permission

正常交易條件：

```text
if danger_score < 0.5
and shock_score < 0.6
and state_confidence > 0.4:
    trade_allowed = true
    reduce_only = false
```

只允許減倉條件：

```text
if danger_score >= 0.75
or shock_score >= 0.8
or p_crash_risk >= 0.5:
    trade_allowed = true
    reduce_only = true
```

---

## 3.12 Rebalance Speed

```text
rebalance_speed =
    0.8
  * state_confidence
  * (1 - transition_risk)
  * liquidity_score
```

限制：

```text
rebalance_speed ∈ [0.05, 0.8]
```

若 shock 過高：

```text
if shock_score > 0.8:
    rebalance_speed = min(rebalance_speed, 0.2)
```

實際曝險調整：

```text
exposure_change =
    rebalance_speed
  * (target_exposure - current_exposure)
```

---

## 3.13 Max Leverage

```text
base_max_leverage = 1.0
```

```text
max_leverage = base_max_leverage * risk_scaler
```

若：

```text
danger_score > 0.75
```

則：

```text
max_leverage = min(max_leverage, 0.3)
```

若：

```text
p_crash_risk > 0.5
```

則：

```text
max_leverage = 0
```

---

## 3.14 C++ 資料結構

```cpp
struct PortfolioStateV1 {
    double current_exposure;      // -1 to 1
    double current_position;      // asset quantity
    double equity;                // account equity
    double cash;                  // cash balance
    double unrealized_pnl;
    double portfolio_drawdown;
    double leverage;
};
```

```cpp
struct ControlActionV1 {
    double target_exposure;       // -1 to 1
    double exposure_change;       // desired exposure adjustment

    double max_leverage;
    double rebalance_speed;

    bool trade_allowed;
    bool reduce_only;

    std::string action_type;
    std::string reason_code;
};
```

---

## 3.15 Layer 3 v1 目標

Layer 3 的目標不是預測價格，而是：

```text
在不同市場狀態下，決定合理的風險曝險。
```

一句話：

> Layer 3 是 exposure controller，不是 alpha model。

---

# 4. Layer 4：風險與穩定性分析 Risk & Stability Analysis v1

## 4.1 定位

Layer 4 的任務是：

> 檢查 Layer 3 產生的控制動作是否安全、穩定、可執行，必要時修正或否決它。

Layer 3 負責「想做什麼」。  
Layer 4 負責「這樣做安不安全」。

---

## 4.2 v1 類型

v1 採用：

```text
Risk & Stability Safety Filter
```

中文：

```text
風險與穩定性安全過濾器
```

---

## 4.3 Input

Layer 4 使用：

```text
MarketStateV1
EstimatedMarketStateV1
PortfolioStateV1
ControlActionV1
RiskConfigV1
```

---

## 4.4 Output

Layer 4 輸出：

```text
SafeControlActionV1
```

定義：

```text
SafeControlActionV1 = {
    safe_target_exposure,
    safe_exposure_change,

    allowed_max_exposure,
    allowed_max_leverage,
    allowed_turnover,

    trade_allowed,
    reduce_only,
    emergency_deleveraging,
    kill_switch,

    final_action_type,
    risk_reason_code
}
```

---

## 4.5 核心風險限制

Layer 4 v1 要保證：

```text
1. 曝險不超過上限
2. 槓桿不超過上限
3. 單次調倉幅度不超過上限
4. 高波動時降低曝險
5. 流動性差時降低交易速度
6. 回撤過大時強制降風險
7. crash risk 太高時 reduce-only 或 kill switch
8. 避免 regime change 期間過度反應
```

---

## 4.6 Allowed Max Exposure

先計算：

```text
vol_penalty = clip((volatility_score - 1.0) / 2.0, 0, 1)
```

```text
dd_severity = clip(abs(portfolio_drawdown) / 0.20, 0, 1)
```

曝險限制：

```text
allowed_max_exposure =
    1.0
  * (1 - danger_score)
  * (1 - 0.6 * vol_penalty)
  * (1 - 0.9 * p_crash_risk)
  * (1 - 0.8 * dd_severity)
```

最後：

```text
allowed_max_exposure = clip(allowed_max_exposure, 0, base_max_exposure)
```

---

## 4.7 Safe Target Exposure

若可多空：

```text
safe_target_exposure =
    clip(
        action.target_exposure,
        -allowed_max_exposure,
        allowed_max_exposure
    )
```

若只做現貨多單：

```text
safe_target_exposure =
    clip(
        action.target_exposure,
        0,
        allowed_max_exposure
    )
```

---

## 4.8 Allowed Turnover

```text
allowed_turnover =
    base_turnover_limit
  * liquidity_score
  * (1 - 0.7 * shock_score)
  * (1 - 0.5 * transition_risk)
```

限制：

```text
allowed_turnover = clip(allowed_turnover, min_turnover, base_turnover_limit)
```

---

## 4.9 Safe Exposure Change

```text
raw_change = safe_target_exposure - current_exposure
```

```text
safe_exposure_change =
    clip(
        raw_change,
        -allowed_turnover,
        allowed_turnover
    )
```

---

## 4.10 Reduce-only 規則

啟動 reduce-only 條件：

```text
danger_score >= 0.75
shock_score >= 0.8
p_crash_risk >= 0.5
portfolio_drawdown <= -0.15
liquidity_score <= 0.25 且 volatility_score 高
```

reduce-only 意思是：

```text
只允許降低曝險，不允許增加曝險。
```

---

## 4.11 Kill Switch 規則

啟動 kill switch 條件：

```text
portfolio_drawdown <= -0.20
danger_score >= 0.90
p_crash_risk >= 0.70
shock_score >= 0.90 and liquidity_score <= 0.20
leverage > allowed_max_leverage * 1.2
```

啟動後：

```text
kill_switch = true
reduce_only = true
emergency_deleveraging = true
safe_target_exposure = 0
```

---

## 4.12 穩定性限制

### 方向翻轉限制

如果目前曝險與目標曝險方向相反：

```text
current_exposure * target_exposure < 0
```

則需要更高信心：

```text
if state_confidence < 0.7 or transition_risk > 0.4:
    target_exposure = 0
```

意思是：

```text
狀態不夠確定時，先降到空手，不要直接反向重倉。
```

---

### 遲滯機制 Hysteresis

```text
if abs(safe_exposure_change) < min_trade_threshold:
    safe_exposure_change = 0
```

目的：

```text
避免因為小幅雜訊造成過度交易。
```

---

## 4.13 RiskConfigV1

```cpp
struct RiskConfigV1 {
    double base_max_exposure = 1.0;
    double base_max_leverage = 1.0;

    double max_allowed_drawdown = 0.20;
    double reduce_only_drawdown = 0.15;
    double kill_switch_drawdown = 0.20;

    double base_turnover_limit = 0.20;
    double min_turnover = 0.01;

    double danger_reduce_threshold = 0.75;
    double danger_kill_threshold = 0.90;

    double shock_reduce_threshold = 0.80;
    double shock_kill_threshold = 0.90;

    double crash_reduce_threshold = 0.50;
    double crash_kill_threshold = 0.70;

    double liquidity_stress_threshold = 0.25;
    double liquidity_kill_threshold = 0.20;

    double min_trade_threshold = 0.03;
};
```

---

## 4.14 SafeControlActionV1

```cpp
struct SafeControlActionV1 {
    double safe_target_exposure;
    double safe_exposure_change;

    double allowed_max_exposure;
    double allowed_max_leverage;
    double allowed_turnover;

    bool trade_allowed;
    bool reduce_only;
    bool emergency_deleveraging;
    bool kill_switch;

    std::string final_action_type;
    std::string risk_reason_code;
};
```

---

## 4.15 Layer 4 v1 目標

Layer 4 的目標不是提高報酬，而是：

```text
避免策略死掉
避免失控
限制尾部風險
限制過度交易
限制高波動下的錯誤放大
限制低流動性下的滑價傷害
```

一句話：

> Layer 4 是整個系統的安全過濾器。

---

# 5. Layer 5：自適應與學習 Adaptation & Learning v1

## 5.1 定位

Layer 5 的任務是：

> 讓系統知道自己什麼時候有效、什麼時候失效，並在市場變化後調整參數。

Layer 5 不直接交易，也不直接下單。  
它負責根據績效、市場狀態與風控紀錄，更新前四層的參數或給出降級建議。

---

## 5.2 v1 類型

v1 採用：

```text
Rule-based Adaptive Supervisor
```

中文：

```text
規則式自適應監督器
```

---

## 5.3 Input

Layer 5 使用完整系統歷史：

```text
SystemHistory = {
    MarketStateV1 history,
    EstimatedMarketStateV1 history,
    ControlActionV1 history,
    SafeControlActionV1 history,
    Portfolio performance history
}
```

重點包含：

```text
strategy_return
equity_curve
realized_pnl
unrealized_pnl
drawdown
max_drawdown
Sharpe ratio
Sortino ratio
win rate
turnover
transaction_cost
slippage
regime-wise performance
Layer 4 intervention rate
```

---

## 5.4 Output

Layer 5 輸出：

```text
AdaptiveUpdateV1
```

定義：

```text
AdaptiveUpdateV1 = {
    model_health_score,
    strategy_health_score,
    overfit_risk_score,

    new_softmax_temperature,
    new_smoothing_alpha,
    new_sideways_penalty,
    new_volatility_penalty,
    new_base_turnover_limit,
    new_danger_reduce_threshold,
    new_danger_kill_threshold,

    adaptation_mode,
    reason_code
}
```

---

## 5.5 主要功能

Layer 5 v1 先做五件事：

```text
1. Performance Monitor 績效監控
2. Model Health Check 模型健康檢查
3. Parameter Adaptation 參數調整
4. Strategy Disable / Reduce Mode 策略降級
5. Walk-forward Learning
```

---

## 5.6 Performance Monitor

監控：

```text
rolling_return
rolling_volatility
rolling_sharpe
rolling_max_drawdown
rolling_turnover
rolling_cost
win_rate
profit_factor
```

還要監控 regime-wise performance：

```text
return_by_regime
drawdown_by_regime
turnover_by_regime
cost_by_regime
```

例如：

```text
bull:       +8.2%
bear:       -1.5%
sideways:   -5.8%
high_vol:   -3.2%
crash_risk: -0.4%
```

---

## 5.7 Model Health Score

```text
model_health_score ∈ [0, 1]
```

考慮因素：

```text
regime_uncertainty
Layer 4 intervention_rate
average_state_confidence
shock frequency
regime rapid flip frequency
```

概念公式：

```text
model_health =
    1.0
  - 0.3 * avg_regime_uncertainty
  - 0.3 * intervention_rate
  - 0.2 * low_confidence_penalty
  - 0.2 * abnormal_shock_frequency
```

---

## 5.8 Strategy Health Score

```text
strategy_health_score ∈ [0, 1]
```

考慮因素：

```text
rolling_sharpe
rolling_drawdown
turnover_cost
intervention_rate
```

概念公式：

```text
strategy_health =
    1.0
  + 0.3 * normalizeSharpe(rolling_sharpe)
  - 0.4 * normalizeDrawdown(rolling_drawdown)
  - 0.2 * normalizeTurnoverCost(turnover, net_return)
  - 0.1 * intervention_rate
```

---

## 5.9 Adaptation Mode

Layer 5 輸出系統模式：

```text
adaptation_mode ∈ {
    normal,
    cautious,
    defensive,
    retrain_required,
    disabled
}
```

| mode | 意義 |
|---|---|
| `normal` | 系統健康，正常運行 |
| `cautious` | 模型或績效略有退化，降低風險 |
| `defensive` | 策略明顯不穩，進入防守 |
| `retrain_required` | 需要重新校正參數 |
| `disabled` | 策略失效或風險過大，暫停策略 |

---

## 5.10 更新規則

### Sideways 下表現差

```text
if return_in_sideways < -threshold
and turnover_in_sideways high:
    increase sideways_penalty
    decrease rebalance_speed
```

---

### High Vol 下回撤太大

```text
if drawdown_in_high_vol > threshold:
    increase volatility_penalty
    lower max_exposure_in_high_vol
```

---

### Crash Risk 下仍虧損過大

```text
if drawdown_in_crash_risk > threshold:
    lower crash_reduce_threshold
    lower crash_kill_threshold
    increase p_crash_risk penalty
```

---

### Turnover 太高但績效不好

```text
if turnover high
and net_return <= 0:
    decrease base_turnover_limit
    increase min_trade_threshold
```

---

### Layer 4 太常否決 Layer 3

```text
if intervention_rate > 0.5:
    reduce Layer 3 base aggressiveness
    increase risk penalty in Layer 3
```

---

### State Confidence 長期偏低

```text
if average_state_confidence < 0.4:
    set adaptation_mode = cautious
    reduce global exposure multiplier
```

---

### 近期績效明顯惡化

```text
if rolling_sharpe < 0
and rolling_drawdown severe:
    set adaptation_mode = defensive
```

如果更嚴重：

```text
if rolling_drawdown exceeds hard limit:
    set adaptation_mode = disabled
```

---

## 5.11 AdaptiveConfigV1

```cpp
struct AdaptiveConfigV1 {
    int performance_window = 100;
    int regime_performance_window = 300;

    double min_model_health = 0.4;
    double min_strategy_health = 0.4;

    double max_param_update_step = 0.05;

    double min_softmax_temperature = 0.5;
    double max_softmax_temperature = 1.5;

    double min_smoothing_alpha = 0.05;
    double max_smoothing_alpha = 0.40;

    double min_base_turnover_limit = 0.02;
    double max_base_turnover_limit = 0.25;

    double max_allowed_parameter_drift = 0.30;
};
```

---

## 5.12 AdaptiveUpdateV1

```cpp
struct AdaptiveUpdateV1 {
    double model_health_score;
    double strategy_health_score;
    double overfit_risk_score;

    double new_softmax_temperature;
    double new_smoothing_alpha;
    double new_sideways_penalty;
    double new_volatility_penalty;
    double new_base_turnover_limit;
    double new_danger_reduce_threshold;
    double new_danger_kill_threshold;

    std::string adaptation_mode;
    std::string reason_code;
};
```

---

## 5.13 Walk-forward Learning

Layer 5 應該使用 walk-forward，而不是一次性最佳化。

流程：

```text
用過去一段資料校正參數
在下一段資料測試
滾動往前
記錄哪些參數穩定有效
拒絕只在某段歷史有效的參數
```

範例：

```text
股票日資料：
training window: 12 months
validation window: 3 months
step size: 1 month
```

```text
BTC 1h 資料：
training window: 90 days
validation window: 14 days
step size: 7 days
```

---

## 5.14 防止 Overfitting 的限制

Layer 5 必須遵守：

```text
1. 更新不能太頻繁
2. 每次更新幅度不能太大
3. 參數有上下限
4. 必須用 walk-forward 驗證
5. 不能用未來資料
6. 不能只因為短期虧損就大改
7. 只更新少數關鍵參數
```

---

## 5.15 Layer 5 v1 目標

Layer 5 的目標不是讓系統無限制自我優化，而是：

```text
讓系統有能力發現自己正在失效，並在受控範圍內調整行為。
```

一句話：

> Layer 5 是整個系統的回饋更新與健康監控層。

---

# 6. v1 完整資料流

```text
1. OHLCV
   ↓
2. Layer 1：MarketStateV1
   - trend_score
   - volatility_score
   - volume_score
   - liquidity_score
   - drawdown
   - shock_score
   - confidence
   - market_mode
   ↓
3. Layer 2：EstimatedMarketStateV1
   - p_bull
   - p_bear
   - p_sideways
   - p_high_vol
   - p_crash_risk
   - state_confidence
   - regime_uncertainty
   - transition_risk
   - danger_score
   ↓
4. Layer 3：ControlActionV1
   - target_exposure
   - exposure_change
   - max_leverage
   - rebalance_speed
   - trade_allowed
   - reduce_only
   ↓
5. Layer 4：SafeControlActionV1
   - safe_target_exposure
   - safe_exposure_change
   - allowed_max_exposure
   - allowed_turnover
   - allowed_max_leverage
   - kill_switch
   ↓
6. Layer 5：AdaptiveUpdateV1
   - model_health_score
   - strategy_health_score
   - adaptation_mode
   - parameter update suggestions
```

---

# 7. v1 的限制

目前 v1 還沒有：

```text
1. Kalman Filter / H-infinity Filter / Particle Filter
2. HMM / Markov Switching 的正式統計估計
3. Robust MPC / Stochastic MPC
4. VaR / CVaR 最佳化
5. 嚴格 Lyapunov stability proof
6. order book execution model
7. 精細交易成本與滑價模型
8. 多資產配置
9. 強化學習
10. 新聞與政策事件 NLP
```

但 v1 已經具備：

```text
1. 市場狀態表示
2. regime belief
3. danger score
4. transition risk
5. 風險感知曝險控制
6. 曝險限制
7. 槓桿限制
8. 單次調倉限制
9. reduce-only
10. kill switch
11. 系統健康監控
12. 自適應更新框架
```

---

# 8. 後續升級方向

未來可以逐層升級：

## Layer 1 升級

```text
加入 order book
加入 spread
加入 funding rate
加入 open interest
加入 macro/news/event features
```

## Layer 2 升級

```text
HMM
Markov Switching Model
Kalman Filter
H-infinity Filter
Particle Filter
Bayesian Online Change Point Detection
IMM Filter
```

## Layer 3 升級

```text
MPC
Robust MPC
Stochastic MPC
Risk-sensitive Control
Online Convex Optimization
Safe Reinforcement Learning
```

## Layer 4 升級

```text
VaR / CVaR constraint
chance constraint
stress testing
robust invariant set
liquidation risk model
transaction cost model
execution risk model
```

## Layer 5 升級

```text
walk-forward optimizer
online learning
meta-parameter learning
regime-specific parameter update
model selection
ensemble controller weighting
```

---

# 9. 總結

目前定案的 v1 系統是：

```text
Layer 1：市場建模
OHLCV → MarketStateV1

Layer 2：狀態估測
MarketStateV1 → EstimatedMarketStateV1

Layer 3：策略控制器
MarketStateV1 + EstimatedMarketStateV1 + PortfolioStateV1 → ControlActionV1

Layer 4：風險與穩定性分析
ControlActionV1 + Risk Constraints → SafeControlActionV1

Layer 5：自適應與學習
SystemHistory + Performance → AdaptiveUpdateV1
```

這個系統的核心不是「預測下一根 K 線」，而是：

> 在不確定、非平穩、有 regime change、有 shock、有流動性風險的市場中，控制我們承擔的風險曝險，並讓系統能夠發現自己何時失效。

