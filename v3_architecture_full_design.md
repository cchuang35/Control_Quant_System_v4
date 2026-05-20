# Control Quant System v3 架構設計文件

> 專案：以控制看量化系統  
> 文件目的：整理 v1、v2 的既有成果，定義 v3 的完整策略架構、控制器分層、資料流、模組邊界、風控邏輯、回測驗證方式，以及 v4 的數學化延伸方向。  
> 目前版本定位：v3 是從 v2 的 BTC 1h wrapper，升級為「長線為主、短線為輔」的分層 exposure controller。  

## Canonical v3 Architecture Requirements

This section is the plain-English canonical architecture summary for v3. If
other sections of this document become stale or hard to read, this section
should be treated as the reference intent.

### v1 Role

v1 is the original five-layer control-system architecture prototype. It
contains the market model, state estimator, strategy controller, risk filter,
and adaptive supervisor concepts that define the initial control-system
framework.

### v2 Role

v2 is a BTCUSDT 1h wrapper around `v1.final`. It does not replace the v1 core;
it wraps the v1 output with a BTC-specific regime gate, binary position logic,
weak-bull losing-trade cooldown, and fee-aware validation.

### v2 Components to Preserve

v3 should preserve the useful lessons from v2:

```text
1. regime gate
2. sideways-hold-only behavior
3. weak_bull losing-trade cooldown
4. fee-aware return calculation
5. multi-fee validation
6. BTC/ETH separated validation
```

These components should be carried forward as design patterns or validation
requirements, not by blindly expanding the v2 wrapper into a larger monolith.

### v3 Goal

v3 should become a long-term-primary, short-term-auxiliary, risk-controlled
exposure controller:

```text
1. long-term primary controller
2. short-term auxiliary controller
3. risk supervisor with highest authority
4. discrete exposure controller
5. fee-aware execution layer
```

The initial v3 exposure set is discrete:

```text
0.00, 0.25, 0.50, 0.75, 1.00
```

### v3 High-Level Flow

```text
Raw OHLCV Data
    -> Data Preprocessor
    -> Feature Builder
    -> Market Estimator v3
    -> Long-term Controller
    -> Short-term Auxiliary Controller
    -> Risk Supervisor
    -> Position Composer
    -> Fee-aware Execution Layer
    -> Backtest / Diagnostics
```

The long-term controller is the primary strategy engine. The short-term
controller is auxiliary and may adjust, delay, or reduce exposure, but it should
not become the main strategy in v3.

### Explicit Deferrals

Particle filter is deferred to v4. v3 should not introduce particle-filter
market estimation.

Short-term-primary architecture is only a future research branch. It is not the
main v3 architecture and should not be mixed into the initial v3 implementation.

---

# 0. 總結：v3 要解決什麼問題？

v1 建立了五層控制式量化系統的骨架。

v2 在 v1.final 外面加上一層小本金、手續費敏感、BTC 1h 專用的 wrapper，核心是：

```text
v1.final signal
    ↓
regime detector
    ↓
trade gate
    ↓
binary position 0 / 1
    ↓
fee-aware backtest
```

v2 證明了幾件事：

1. 不能在所有市場狀態下交易。
2. regime gate 對風控有幫助。
3. weak_bull 狀態下連續失敗後，暫停新進場可以降低錯誤交易。
4. 小本金策略一定要考慮交易成本。
5. BTC 和 ETH 的驗證結果不能混在一起看；同一個策略不一定跨資產有效。

但是 v2 還不是完整的分層控制系統。

v2 的限制是：

```text
1. 主要是 BTCUSDT 1h 專用，不是 general strategy。
2. 倉位是 binary：0 或 1。
3. 沒有明確分成長線主控制器與短線輔助控制器。
4. 風控主要是 regime gate + weak_bull cooldown，還不是完整 Risk Supervisor。
5. 仍然保留 v1.final 為核心，v2 只是外層 wrapper。
6. market estimation 仍是 MA / momentum / volatility 規則，還沒有數學化 estimator。
```

因此 v3 的核心目標是：

> **把 v1 的五層骨架與 v2 的有效風控 wrapper，重構成「長線為主、短線為輔、風控最高權限、離散倉位、低換手率」的分層 exposure controller。**

v3 暫時不急著導入 particle filter。  
v3 的主線是架構重構、策略分層、倉位控制、風控模組化。  
v4 才開始導入 particle filter、uncertainty、regime probability 等數學化 market estimation。

---

# 1. 版本定位

## 1.1 v1.final 的定位

v1.final 是目前系統的保守基準策略。

它的重點不是追求最高報酬，而是建立一個可解釋、可逐層升級的控制式交易系統。

v1 的五層架構：

```text
Layer 1：Market Modeling
Layer 2：State Estimation
Layer 3：Strategy Controller
Layer 4：Risk & Stability Analysis
Layer 5：Adaptation & Learning
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

v1 的價值：

```text
1. 建立控制系統的基本分層。
2. 將市場資料轉成狀態變數。
3. 將狀態變數轉成控制動作。
4. 將風控獨立成一層。
5. 保留未來自適應與學習空間。
```

v1 的限制：

```text
1. 策略行為仍偏單層。
2. 長線與短線沒有明確分工。
3. 倉位控制還不夠精細。
4. 風控與交易成本尚未完全以小本金場景為核心設計。
5. 尚未明確處理多資產驗證與 regime-specific validation。
```

---

## 1.2 v2 的定位

v2 是在 v1.final 外面加上的小本金風控 wrapper。

v2 的核心不是重寫 v1，而是：

```text
保留 v1.final
+ regime detector
+ trade gate
+ weak_bull cooldown
+ binary position
+ fee-aware backtest
+ validation diagnostics
```

v2 的代表版本：

```text
v2.btc_final_candidate_A
```

其核心規則：

```text
strong_bull:
    allow_entry = true
    allow_hold = true

weak_bull:
    allow_entry = true by default
    allow_hold = true
    but weak_bull new entries are blocked during weak_bull_loss_cooldown

sideways:
    allow_entry = false
    allow_hold = true

bear:
    allow_entry = false
    allow_hold = false
```

weak_bull losing-trade cooldown：

```text
If a completed trade has:
    entry_regime == weak_bull
    and net_trade_return < 0
then block new weak_bull entries for cooldown_bars.
```

預設：

```text
cooldown_bars = 120
observed robust range = 120 / 144 / 168 bars
interpretation = roughly 5-7 days on 1h bars
```

v2 的倉位：

```text
position ∈ {0, 1}
```

也就是：

```text
0 = 空手
1 = 滿倉
```

v2 的 fee-aware return：

```text
strategy_return_net[t]
=
position[t-1] * asset_return[t]
-
abs(position[t] - position[t-1]) * fee_rate
```

v2 的價值：

```text
1. 驗證 regime gate 有用。
2. 驗證 sideways-hold-only 比粗暴退出更細緻。
3. 驗證 weak_bull losing-trade cooldown 是有效的短期風控記憶。
4. 驗證 fee-aware backtest 對小本金非常重要。
5. 驗證 BTC 與 ETH 不能直接共用同一套策略結論。
6. 建立多 fee_rate、多資料區間、多 cooldown 參數的 robustness 思維。
```

v2 的限制：

```text
1. 目前是 BTCUSDT 1h specific candidate。
2. ETH 驗證沒有通過。
3. 不是 general cross-asset final strategy。
4. 倉位是 binary，還不是真正 exposure controller。
5. 沒有明確長線主、短線輔架構。
6. Market Estimator 還是簡單規則，不是數學化 state estimation。
```

---

## 1.3 v3 的定位

v3 的定位是：

> **Long-term Primary, Short-term Auxiliary, Risk-Controlled Exposure Controller**

中文：

> **長線主導、短線輔助的風險控制倉位控制器**

v3 不再只是 v1.final 的外層 binary trade gate。  
v3 要把策略提升成分層控制系統。

v3 的核心資料流：

```text
Market Data
    ↓
Market Feature Builder
    ↓
Market Estimator v3
    ↓
Long-term Controller
    ↓
Short-term Auxiliary Controller
    ↓
Risk Supervisor
    ↓
Fee-aware Execution Layer
    ↓
Final Position
    ↓
Backtest / Diagnostics / Validation
```

v3 的核心原則：

```text
1. 長線系統決定主要方向與基礎倉位。
2. 短線系統只做輔助加減碼，不直接推翻長線方向。
3. Risk Supervisor 擁有最高權限。
4. 小本金優先，低換手率優先。
5. 倉位先使用離散倉位。
6. 槓桿只在高信心、低風險、低不確定性狀態下考慮。
7. 策略評分以最大回撤與 Sharpe ratio 為第一優先。
8. 年化報酬是重要但不是最高優先。
9. 不追求高勝率，而是追求風險調整後報酬與可存活性。
```

---

## 1.4 v4 的定位

v4 是 v3 完成後的數學化版本。

v4 的主題：

> **Mathematical Market Estimation and Uncertainty-aware Control**

v4 會考慮導入：

```text
1. particle filter
2. hidden trend estimation
3. uncertainty estimation
4. regime probability
5. dynamic controller gain
6. uncertainty-aware leverage
7. state-space model
8. adaptive risk cap
```

v4 不應該在 v3 架構穩定前開始，否則系統會同時有太多變數，無法判斷哪個模組真的有效。

---

# 2. 使用者策略條件與設計限制

目前使用者條件：

```text
資金規模：約新台幣 1 萬至 4 萬
主要標的：美股或加密貨幣，尤其 BTC / ETH 類型
風格偏好：風險小，但不希望報酬太低
最大回撤：可接受約 -20% 等級，但必須有更好的整體收益作為補償
交易頻率：可分長線與短線，但主架構先長線為主
倉位形式：先離散，之後再考慮連續
槓桿：可研究，但只能在高信心、低風險時使用
評估指標：最大回撤 = Sharpe ratio > 年化報酬 > 勝率 > 穩定性
```

因此 v3 不適合一開始做：

```text
1. 高頻交易
2. 日內主導策略
3. 連續倉位每日微調
4. 高槓桿 all-in
5. 純 mean-reversion 抄底
6. 只看勝率的策略
7. 過多技術指標投票
8. 沒有成本模型的短線回測
```

v3 適合做：

```text
1. 長線趨勢主導
2. 短線輔助加減碼
3. 低換手率
4. 離散倉位
5. fee-aware execution
6. drawdown-aware risk cap
7. regime-specific behavior
8. BTC / ETH / 美股分開驗證
9. 逐步加入可替換的 estimator
```

---

# 3. v3 核心策略定義

v3 策略名稱：

```text
Risk-Controlled Dual-Horizon Trend Strategy
```

中文：

```text
風險控制雙週期趨勢策略
```

核心一句話：

> **長線決定主要方向，短線決定局部調整，風控決定最大允許倉位，交易層決定是否值得執行。**

v3 不是單純回答：

```text
現在要不要買？
```

而是回答：

```text
1. 現在市場大方向是什麼？
2. 長線應該持有多少基礎倉位？
3. 短線是否提供加碼、減碼或暫停訊號？
4. 目前風險最多允許多少倉位？
5. 交易成本是否值得這次調整？
6. 最終倉位應該是多少？
```

---

# 4. v3 與 v2 的關係

v3 不是拋棄 v2，而是重構 v2。

v2 的東西在 v3 中會被重新分類。

| v2 元件 | v3 中的位置 | 說明 |
|---|---|---|
| MA20 / MA60 / momentum / volatility regime detector | Market Estimator v3 初版 | 先保留規則型 estimator |
| strong_bull / weak_bull / sideways / bear | Long-term / regime state | 可作為長線狀態初版 |
| sideways-hold-only | Short-term permission rule | 震盪時不開新倉，但可視情況續抱 |
| weak_bull loss cooldown | Short-term risk memory | 弱多頭失敗後暫停弱多進場或加碼 |
| fee-aware return | Backtest / Execution Layer | 保留並擴充交易成本模型 |
| binary position | v3 初期相容模式 | 但 v3 主線改成離散 exposure |
| BTC-specific validation | Asset-specific validation | 不假設跨資產有效 |

---

# 5. v3 總架構

## 5.1 高層架構

```text
Raw OHLCV Data
    ↓
Data Preprocessor
    ↓
Feature Builder
    ↓
Market Estimator v3
    ↓
Long-term Controller
    ↓
Short-term Auxiliary Controller
    ↓
Risk Supervisor
    ↓
Position Composer
    ↓
Fee-aware Execution Layer
    ↓
Backtest Engine
    ↓
Diagnostics / Validation / Reports
```

---

## 5.2 控制系統對應關係

傳統控制系統：

```text
Plant → Sensor → State Estimator → Controller → Actuator → Feedback
```

交易系統對應：

```text
Market → OHLCV Data → Market Estimator → Exposure Controller → Trade Execution → Portfolio Feedback
```

v3 的控制意義：

```text
Market = plant
OHLCV = sensor measurement
Market Estimator = observer / state estimator
Long-term Controller = strategic controller
Short-term Controller = tactical controller
Risk Supervisor = safety constraint layer
Execution Layer = actuator with transaction cost
Backtest Metrics = feedback signal
```

---

# 6. v3 模組詳細設計

---

# 6.1 Data Preprocessor

## 6.1.1 目的

Data Preprocessor 負責整理原始資料，確保後續模組拿到乾淨、對齊、無 look-ahead bias 的資料。

## 6.1.2 Input

```text
timestamp
open
high
low
close
volume
```

未來可加入：

```text
funding rate
open interest
order book proxy
macro data
index data
market breadth
```

但 v3 初版不需要。

## 6.1.3 Output

```text
clean_ohlcv
```

包含：

```text
1. timestamp sorted
2. duplicate removed
3. missing bars flagged
4. returns calculated
5. asset-specific calendar handled
6. no future data leakage
```

## 6.1.4 注意事項

對加密貨幣：

```text
annualization_factor = 365 或 365 * 24，視資料頻率而定
```

對股票：

```text
annualization_factor = 252，若使用日線
```

對 1h 資料：

```text
crypto annual bars ≈ 365 * 24
stock annual bars ≈ 252 * 每日交易小時數
```

---

# 6.2 Feature Builder

## 6.2.1 目的

Feature Builder 負責把 raw OHLCV 轉成 estimator 與 controller 可使用的 feature。

## 6.2.2 基礎 features

v3 初版建議保留 v1 / v2 已使用過的特徵：

```text
return_1
log_return
rolling_volatility
volatility_score
MA_short
MA_long
trend_raw
trend_score
momentum_n
drawdown
volume_z
volume_score
range_ratio
liquidity_proxy
shock_score
```

## 6.2.3 長線 features

長線 features 用於 Long-term Controller。

若資料是日線，可使用：

```text
MA50
MA100
MA200
momentum_60
momentum_120
momentum_180
volatility_60
drawdown_120
drawdown_252
```

若資料是 1h，可使用較長 bars 近似週期：

```text
MA_24h
MA_7d
MA_30d
momentum_7d
momentum_30d
volatility_7d
volatility_30d
drawdown_30d
```

## 6.2.4 短線 features

短線 features 用於 Short-term Auxiliary Controller。

可使用：

```text
short_momentum
short_pullback
short_overheat
short_breakdown
short_recovery
intraday_volatility
recent_return_zscore
recent_drawdown
```

## 6.2.5 風控 features

Risk Supervisor 使用：

```text
portfolio_drawdown
asset_drawdown
realized_volatility
volatility_spike
shock_score
turnover
fee_drag
consecutive_losses
cooldown_state
```

---

# 6.3 Market Estimator v3

## 6.3.1 目的

Market Estimator v3 的任務是將 feature 轉成市場狀態。

v3 初版先使用規則型 estimator，不急著使用 particle filter。

## 6.3.2 Input

```text
features_t
```

## 6.3.3 Output

```text
MarketEstimateV3_t = {
    long_regime,
    short_regime,
    trend_strength,
    volatility_state,
    drawdown_state,
    risk_state,
    confidence_score,
    entry_permission,
    hold_permission
}
```

## 6.3.4 long_regime

長線 regime 可分成：

```text
strong_bull
bull
neutral
bear
strong_bear
```

這是 v2 的 strong_bull / weak_bull / sideways / bear 的升級版。

v2：

```text
strong_bull
weak_bull
sideways
bear
```

v3：

```text
strong_bull
bull
neutral
bear
strong_bear
```

其中：

```text
v2 strong_bull → v3 strong_bull
v2 weak_bull   → v3 bull 或 weak_bull 子狀態
v2 sideways    → v3 neutral
v2 bear        → v3 bear / strong_bear
```

## 6.3.5 short_regime

短線 regime 可分成：

```text
pullback
recovery
overheat
breakdown
noise
```

解釋：

| short_regime | 意義 | 可能行為 |
|---|---|---|
| pullback | 長線仍好，但短線回檔 | 可加碼 |
| recovery | 短線從弱轉強 | 可小幅加碼 |
| overheat | 短線漲太快 | 可減碼 |
| breakdown | 短線跌破重要結構 | 可減碼 |
| noise | 沒明確訊號 | 不動 |

## 6.3.6 confidence_score

confidence_score 不應只等於勝率。

它應該綜合：

```text
trend_strength
volatility_state
regime_consistency
recent_signal_quality
cooldown_state
fee_environment
```

初版可定義：

```text
confidence_score ∈ [0, 1]
```

解讀：

```text
0.0 ~ 0.3：低信心
0.3 ~ 0.6：中等信心
0.6 ~ 0.8：高信心
0.8 ~ 1.0：極高信心
```

v3 中，confidence_score 用於：

```text
1. 是否允許加碼
2. 是否允許槓桿
3. 是否放寬 risk cap
4. 是否啟動 no-trade zone
```

---

# 6.4 Long-term Controller

## 6.4.1 目的

Long-term Controller 是 v3 的主控制器。

它負責決定：

```text
base_position
```

也就是主要倉位。

v3 的核心決議是：

> **長線為主，短線為輔。**

所以長線控制器的輸出權重最大。

## 6.4.2 Input

```text
long_regime
trend_strength
volatility_state
drawdown_state
confidence_score
```

## 6.4.3 Output

```text
base_position
```

初版離散倉位：

```text
0%, 25%, 50%, 75%, 100%
```

可選槓桿版本：

```text
0%, 25%, 50%, 75%, 100%, 125%
```

## 6.4.4 基本規則

```text
strong_bull  → base_position = 75% 或 100%
bull         → base_position = 50% 或 75%
neutral      → base_position = 25% 或 50%
bear         → base_position = 0% 或 25%
strong_bear  → base_position = 0%
```

較保守版本：

```text
strong_bull  → 75%
bull         → 50%
neutral      → 25%
bear         → 0%
strong_bear  → 0%
```

較積極版本：

```text
strong_bull  → 100%
bull         → 75%
neutral      → 50%
bear         → 25%
strong_bear  → 0%
```

v3 初版建議使用保守版本或中間版本。

## 6.4.5 與 v2 的差異

v2：

```text
regime gate 決定是否允許 v1 signal 進場
position = 0 or 1
```

v3：

```text
long_regime 決定 base_position
position 可以是 0 / 25 / 50 / 75 / 100
```

也就是 v3 不只是問：

```text
要不要持有？
```

而是問：

```text
應該持有多少？
```

---

# 6.5 Short-term Auxiliary Controller

## 6.5.1 目的

Short-term Auxiliary Controller 是輔助控制器，不是主控制器。

它負責：

```text
position_adjustment
```

也就是在 base_position 的基礎上加減碼。

## 6.5.2 核心限制

短線控制器不能直接推翻長線控制器。

例如：

```text
長線 strong_bull，短線 overheat：
    可以減碼，但不應直接變成 0%，除非 Risk Supervisor 要求。

長線 strong_bear，短線 recovery：
    可以觀察或小幅試單，但不應直接滿倉。
```

## 6.5.3 Input

```text
short_regime
long_regime
trend_strength
recent_return
recent_drawdown
cooldown_state
confidence_score
```

## 6.5.4 Output

```text
position_adjustment ∈ {-25%, 0%, +25%}
```

v3 初版不建議讓短線調整超過 25%。

## 6.5.5 基本規則

```text
if long_regime in {strong_bull, bull}:
    if short_regime == pullback:
        adjustment = +25%
    elif short_regime == overheat:
        adjustment = -25%
    elif short_regime == breakdown:
        adjustment = -25%
    else:
        adjustment = 0%

elif long_regime == neutral:
    if short_regime == recovery and confidence_score high:
        adjustment = +25%
    elif short_regime == breakdown:
        adjustment = -25%
    else:
        adjustment = 0%

elif long_regime in {bear, strong_bear}:
    if short_regime == recovery and confidence_score very high:
        adjustment = +0% or +25% only in experimental mode
    else:
        adjustment = 0% or -25%
```

## 6.5.6 將 v2 weak_bull cooldown 放進 v3

v2 的 weak_bull losing-trade cooldown 在 v3 中應該保留。

v3 解釋：

> 當市場處於弱多頭或不穩定多頭狀態，且最近一次在該狀態下進場失敗，代表目前此 regime 的訊號可靠度下降，因此短期內降低該 regime 的進場權重。

v3 實作：

```text
if long_regime == bull or weak_bull_substate == true:
    if weak_bull_loss_cooldown_active:
        block_new_addition = true
        adjustment = min(adjustment, 0)
```

也就是：

```text
cooldown active 時：
    不允許 weak_bull 加碼
    不一定強制退出既有部位
    strong_bull 若條件足夠仍可允許進場
```

這保留 v2 的核心優點。

---

# 6.6 Risk Supervisor

## 6.6.1 目的

Risk Supervisor 是最高權限層。

它的任務不是產生 alpha，而是確保系統不要因為策略錯誤、連續虧損、波動失控或槓桿誤用而爆掉。

使用者的目標排序：

```text
最大回撤 = Sharpe ratio > 年化報酬 > 勝率 > 穩定性
```

因此 v3 的 Risk Supervisor 非常重要。

## 6.6.2 Input

```text
portfolio_drawdown
asset_drawdown
realized_volatility
volatility_spike
shock_score
current_position
base_position
position_adjustment
confidence_score
recent_trade_results
fee_drag
turnover
```

## 6.6.3 Output

```text
risk_cap
risk_action
```

risk_cap：

```text
最大允許倉位
```

risk_action：

```text
normal
reduce_only
no_new_entry
force_deleverage
risk_off
```

## 6.6.4 Drawdown-based risk cap

初版可使用：

```text
portfolio_drawdown > -5%:
    risk_cap = 100%

portfolio_drawdown <= -5%:
    risk_cap = 75%

portfolio_drawdown <= -10%:
    risk_cap = 50%

portfolio_drawdown <= -15%:
    risk_cap = 25%

portfolio_drawdown <= -20%:
    risk_cap = 0% or 25%
```

注意：

> 使用者可接受 -20% 左右最大回撤，不代表系統要等到 -20% 才反應。  
> v3 應該在 -10% 附近就開始明顯降風險。

## 6.6.5 Volatility-based risk cap

當波動失控時，降低 risk_cap。

例如：

```text
if realized_volatility > normal_volatility * 1.5:
    risk_cap = min(risk_cap, 75%)

if realized_volatility > normal_volatility * 2.0:
    risk_cap = min(risk_cap, 50%)

if realized_volatility > normal_volatility * 2.5:
    risk_cap = min(risk_cap, 25%)
```

## 6.6.6 Shock-based rule

若單根 K 或短期跌幅過大：

```text
if shock_score is extreme:
    block_new_entry = true
    allow_reduce = true
```

但不一定要強制全部退出，因為強制退出可能賣在低點。  
是否退出要看 long_regime 與 risk_state。

## 6.6.7 Consecutive loss rule

連續虧損後降低交易頻率：

```text
if consecutive_losses >= 2:
    block_new_entry = true for short-term additions

if consecutive_losses >= 3:
    reduce risk_cap

if consecutive_losses >= 4:
    risk_off or paper-trade-only mode
```

v2 的 weak_bull loss cooldown 是這個概念的 regime-specific 版本。

---

# 6.7 Position Composer

## 6.7.1 目的

Position Composer 將長線、短線、風控結果整合成 preliminary target position。

## 6.7.2 基本公式

```text
raw_target_position = base_position + position_adjustment
```

再套用風控：

```text
risk_limited_position = min(raw_target_position, risk_cap)
```

再套用上下限：

```text
risk_limited_position = clip(risk_limited_position, 0%, max_position)
```

若 v3 初版不放空：

```text
min_position = 0%
```

若未來允許放空：

```text
min_position = -25% or -50%
```

但 v3 初版不建議放空。

## 6.7.3 離散化

v3 初版使用：

```text
allowed_positions = [0%, 25%, 50%, 75%, 100%]
```

可選：

```text
allowed_positions = [0%, 25%, 50%, 75%, 100%, 125%]
```

離散化方法：

```text
final_target_position = nearest_allowed_position(risk_limited_position)
```

或保守一點：

```text
final_target_position = floor_to_allowed_position(risk_limited_position)
```

v3 初版建議用保守 floor，而不是 nearest，避免不小心放大風險。

---

# 6.8 Fee-aware Execution Layer

## 6.8.1 目的

Execution Layer 決定是否真的執行 target_position。

因為使用者本金不大，交易成本非常重要。

v3 不能只看理論 target position，還要問：

> 這次調倉的預期好處是否大於交易成本？

## 6.8.2 Input

```text
current_position
target_position
fee_rate
estimated_slippage
spread_cost
minimum_trade_size
confidence_score
risk_action
```

## 6.8.3 Output

```text
executed_position
trade_amount
execution_decision
```

## 6.8.4 No-trade zone

v3 初版規則：

```text
if abs(target_position - current_position) < 25%:
    executed_position = current_position
```

因為倉位本身是 25% 間隔，所以小於 25% 的調整沒有必要。

## 6.8.5 Fee-aware rule

若要更進一步：

```text
expected_edge = confidence_score * expected_return_estimate
expected_cost = abs(target_position - current_position) * fee_rate + slippage

if expected_edge <= expected_cost * cost_safety_multiplier:
    do_not_trade
```

v3 初版可以先不用 expected_edge，只保留 no-trade zone 與 fee-aware backtest。

## 6.8.6 Return 計算

保留 v2 的核心計算：

```text
strategy_return_net[t]
=
position[t-1] * asset_return[t]
-
abs(position[t] - position[t-1]) * fee_rate
```

若加入滑價：

```text
strategy_return_net[t]
=
position[t-1] * asset_return[t]
-
abs(position[t] - position[t-1]) * fee_rate
-
abs(position[t] - position[t-1]) * slippage_rate
```

---

# 7. 大跌時的 v3 決策規則

使用者偏好：

```text
大跌時可能分批加碼，也可能分段降倉。
如果系統判斷未來較可能漲，可以加碼。
如果不確定或覺得會繼續跌，就降倉。
```

因此 v3 不能固定成「跌就買」或「跌就賣」。

v3 必須先看長線狀態。

---

## 7.1 情境 1：長線多頭，短線回檔

條件：

```text
long_regime in {strong_bull, bull}
short_regime == pullback
volatility_state not extreme
portfolio_drawdown acceptable
confidence_score medium/high
```

行為：

```text
允許分批加碼
```

例如：

```text
base_position = 50%
adjustment = +25%
risk_cap = 100%
final_position = 75%
```

或：

```text
base_position = 75%
adjustment = +25%
risk_cap = 100%
final_position = 100%
```

---

## 7.2 情境 2：長線不明，短線大跌

條件：

```text
long_regime == neutral
short_regime == breakdown or pullback
volatility_state rising
confidence_score low/medium
```

行為：

```text
不急著加碼
維持低倉位或降低倉位
```

例如：

```text
base_position = 25%
adjustment = 0% or -25%
risk_cap = 50%
final_position = 0% or 25%
```

---

## 7.3 情境 3：長線空頭，短線大跌

條件：

```text
long_regime in {bear, strong_bear}
short_regime == breakdown
volatility_state high/extreme
drawdown expanding
```

行為：

```text
避免抄底
保持低倉位或空手
```

例如：

```text
base_position = 0%
adjustment = 0%
risk_cap = 25%
final_position = 0%
```

---

## 7.4 情境 4：長線空頭，但短線 recovery

這是最容易誤判的情境。

條件：

```text
long_regime in {bear, strong_bear}
short_regime == recovery
confidence_score high
```

v3 初版建議：

```text
不要直接大幅加碼
最多進入 observation mode 或 small test position
```

例如：

```text
base_position = 0%
adjustment = +0% or +25% only in experimental mode
risk_cap = 25%
final_position = 0% or 25%
```

---

# 8. 槓桿規則

v3 可以保留槓桿研究空間，但不應讓槓桿成為初版主軸。

## 8.1 為什麼不能只看勝率？

勝率高不代表策略好。

例如：

```text
勝率 80%
每次賺 +0.3%
每次輸 -2.0%
```

這種策略可能長期仍然不好。

槓桿條件應該看：

```text
1. trend_strength 高
2. confidence_score 高
3. volatility_state 不高
4. portfolio_drawdown 不深
5. recent trades 沒有連續失敗
6. fee_drag 不高
7. expected return / expected risk 足夠好
```

## 8.2 v3 槓桿上限

v3 初版建議：

```text
default max_position = 100%
optional high-confidence max_position = 125%
```

不建議 v3 初版超過：

```text
150%
```

## 8.3 槓桿啟用條件

```text
if long_regime == strong_bull
and confidence_score >= 0.8
and volatility_state in {low, normal}
and portfolio_drawdown > -5%
and weak_bull_cooldown is not active
and recent_consecutive_losses == 0:
    allow_leverage = true
else:
    allow_leverage = false
```

槓桿 target：

```text
if allow_leverage:
    max_position = 125%
else:
    max_position = 100%
```

---

# 9. v3 Objective Function

v3 的目標不是最大化報酬，而是在風險受控下追求合理報酬。

使用者排序：

```text
最大回撤 = Sharpe ratio > 年化報酬 > 勝率 > 穩定性
```

因此 v3 的 scoring function 可以設計為：

```text
Score
=
+ w_return * annual_return
+ w_sharpe * sharpe_ratio
- w_drawdown * max_drawdown_penalty
- w_turnover * turnover_penalty
- w_fee * fee_drag_penalty
- w_instability * instability_penalty
```

其中：

```text
w_drawdown 高
w_sharpe 高
w_return 中等
w_turnover 中等
w_fee 中等或偏高
w_instability 中等
```

範例：

```text
w_drawdown = 3.0
w_sharpe = 3.0
w_return = 1.5
w_turnover = 1.0
w_fee = 1.0
w_instability = 0.5
```

注意：

> 這不是唯一公式，而是用來比較版本的內部 ranking。  
> v3 不應該只用單一總分，仍應分別看 annual return、max drawdown、Sharpe、turnover、fee drag。

---

# 10. v3 回測與驗證標準

## 10.1 必看指標

v3 每個版本都要輸出：

```text
total_return
annual_return
max_drawdown
sharpe_ratio
sortino_ratio, optional
win_rate
profit_factor
number_of_trades
average_holding_period
turnover
fee_drag
exposure_mean
exposure_max
exposure_distribution
regime_distribution
regime_performance
```

## 10.2 regime-specific diagnostics

必須檢查：

```text
strong_bull 中的報酬
bull 中的報酬
neutral 中的報酬
bear 中的報酬
strong_bear 中的報酬
```

以及：

```text
pullback adjustment 是否有效
overheat reduction 是否有效
breakdown reduction 是否有效
weak_bull cooldown 是否有效
sideways-hold-only 是否仍然有價值
```

## 10.3 fee sensitivity

沿用 v2 的精神，至少測：

```text
fee_rate = 0.0005
fee_rate = 0.0010
fee_rate = 0.0020
```

若是美股，另設股票版本成本模型。

## 10.4 rolling validation

不能只看 full-period。

要看 rolling windows：

```text
rolling 90d
rolling 180d
rolling 365d
rolling 2y, if enough data
```

檢查：

```text
rolling Sharpe
rolling max drawdown
rolling total return
rolling turnover
rolling exposure
```

## 10.5 cross-asset validation

v2 的教訓是：

> BTC 有效，不代表 ETH 有效。

v3 驗證時要分開：

```text
BTCUSDT
ETHUSDT
SPY or QQQ
optional: 台股 ETF
```

每個資產可以有 asset-specific config，但不能偷偷對每個資產過度調參。

## 10.6 out-of-sample validation

至少分成：

```text
train / design period
validation period
out-of-sample period
```

不要在同一段資料上反覆調參後說策略有效。

---

# 11. v3 實作建議：檔案結構

建議新建或重構成：

```text
src/
  data/
    preprocessor.py
    feature_builder.py

  estimator/
    market_estimator_v3.py
    regime_detector.py

  controllers/
    long_term_controller.py
    short_term_aux_controller.py
    position_composer.py

  risk/
    risk_supervisor.py
    cooldown_manager.py
    drawdown_brake.py

  execution/
    fee_model.py
    execution_layer.py

  backtest/
    backtest_engine.py
    metrics.py
    diagnostics.py

  configs/
    v3_default.yaml
    v3_btc_1h.yaml
    v3_eth_1h.yaml
    v3_spy_daily.yaml

  reports/
    report_generator.py
```

如果要保留目前 v1 五層檔案，也可以先用相容方式：

```text
src/layer1_market_model.py
src/layer2_state_estimator.py
src/layer3_strategy_controller.py
src/layer4_risk_filter.py
src/layer5_adaptive_supervisor.py
src/backtest.py
```

但 v3 建議額外加：

```text
src/v3_market_estimator.py
src/v3_long_term_controller.py
src/v3_short_term_controller.py
src/v3_risk_supervisor.py
src/v3_execution_layer.py
src/v3_backtest.py
```

這樣比較不會破壞 v1 / v2。

---

# 12. v3 Data Classes 建議

## 12.1 MarketFeaturesV3

```python
@dataclass
class MarketFeaturesV3:
    timestamp: Any
    close: float
    return_1: float
    log_return: float

    ma_short: float
    ma_long: float
    ma_long_term: float

    momentum_short: float
    momentum_medium: float
    momentum_long: float

    volatility_short: float
    volatility_long: float
    volatility_ratio: float

    drawdown_short: float
    drawdown_long: float

    volume_z: float | None = None
    liquidity_score: float | None = None
    shock_score: float | None = None
```

## 12.2 MarketEstimateV3

```python
@dataclass
class MarketEstimateV3:
    timestamp: Any
    long_regime: str
    short_regime: str
    trend_strength: float
    volatility_state: str
    drawdown_state: str
    risk_state: str
    confidence_score: float

    allow_entry: bool
    allow_hold: bool
    notes: dict
```

## 12.3 LongTermDecisionV3

```python
@dataclass
class LongTermDecisionV3:
    timestamp: Any
    base_position: float
    reason: str
    long_regime: str
    confidence_score: float
```

## 12.4 ShortTermDecisionV3

```python
@dataclass
class ShortTermDecisionV3:
    timestamp: Any
    position_adjustment: float
    reason: str
    short_regime: str
    cooldown_active: bool
```

## 12.5 RiskDecisionV3

```python
@dataclass
class RiskDecisionV3:
    timestamp: Any
    risk_cap: float
    risk_action: str
    reason: str
    portfolio_drawdown: float
    realized_volatility: float
```

## 12.6 FinalPositionDecisionV3

```python
@dataclass
class FinalPositionDecisionV3:
    timestamp: Any
    base_position: float
    position_adjustment: float
    raw_target_position: float
    risk_cap: float
    target_position: float
    executed_position: float
    trade_amount: float
    execution_reason: str
```

---

# 13. v3 主流程 Pseudocode

```python
for each bar t:
    features_t = feature_builder.update(ohlcv_history_until_t)

    estimate_t = market_estimator.estimate(features_t)

    long_decision_t = long_term_controller.decide(
        estimate_t,
        portfolio_state_t,
    )

    short_decision_t = short_term_controller.decide(
        estimate_t,
        portfolio_state_t,
        cooldown_state_t,
    )

    risk_decision_t = risk_supervisor.evaluate(
        estimate_t,
        portfolio_state_t,
        long_decision_t,
        short_decision_t,
    )

    target_position_t = position_composer.compose(
        base_position=long_decision_t.base_position,
        adjustment=short_decision_t.position_adjustment,
        risk_cap=risk_decision_t.risk_cap,
    )

    executed_position_t = execution_layer.decide(
        current_position=current_position_t,
        target_position=target_position_t,
        fee_rate=fee_rate,
        confidence_score=estimate_t.confidence_score,
    )

    # avoid look-ahead:
    # signal at t determines position at t+1
    position_for_next_bar = executed_position_t

    update cooldown manager after trade closed
    update portfolio state after return realized
```

---

# 14. Look-ahead Bias 規則

v3 必須延續 v2 的正確設計：

```text
Signal at bar t determines position for bar t+1.
Return at bar t is earned by position from t-1.
```

錯誤寫法：

```text
position[t] * return[t]
```

可能造成 look-ahead bias。

正確寫法：

```text
strategy_return[t] = position[t-1] * asset_return[t] - trade_cost[t]
```

其中：

```text
trade_cost[t] = abs(position[t] - position[t-1]) * fee_rate
```

---

# 15. v3 開發路線

## 15.1 v3.1：架構重構版

目標：

```text
把 v2 wrapper 重構成 v3 分層架構。
```

實作：

```text
Market Estimator v3 初版
Long-term Controller
Short-term Auxiliary Controller
Risk Supervisor 初版
Position Composer
Fee-aware Execution Layer
```

暫時不做：

```text
particle filter
continuous position
high leverage
short-selling
complex optimization
```

---

## 15.2 v3.2：離散倉位版

目標：

```text
從 binary 0/1 升級成 0/25/50/75/100。
```

要比較：

```text
v2 binary wrapper
vs
v3 discrete exposure controller
```

檢查：

```text
max drawdown 是否下降
Sharpe 是否上升
turnover 是否可接受
fee drag 是否可接受
return 是否沒有被壓太低
```

---

## 15.3 v3.3：短線輔助控制器版

目標：

```text
加入 pullback / overheat / breakdown / recovery 的短線調整。
```

限制：

```text
short adjustment ∈ {-25%, 0%, +25%}
```

檢查：

```text
短線輔助是否真的改善進出場
是否增加過多交易
是否降低 Sharpe
是否提高 fee drag
```

---

## 15.4 v3.4：Risk Supervisor 強化版

目標：

```text
加入 drawdown cap、volatility cap、consecutive loss rule。
```

檢查：

```text
max drawdown 是否下降
是否過度降低年化報酬
是否改善 rolling Sharpe
是否減少 crash period loss
```

---

## 15.5 v3.5：槓桿限制研究版

目標：

```text
只在高信心、低波動、低 drawdown 狀態下允許 125% exposure。
```

限制：

```text
max_position = 125%
不做 150% 以上
不做短線槓桿主導
```

檢查：

```text
槓桿是否真的提升風險調整後報酬
是否讓 max drawdown 明顯惡化
是否只在少數高品質區間啟用
```

---

# 16. v4 延伸方向

v4 在 v3 穩定後才開始。

## 16.1 Particle Filter 的定位

Particle filter 不只是平滑價格，而是 market state estimation。

它估計的是 hidden state，例如：

```text
hidden_trend
hidden_volatility
risk_state
regime_probability
uncertainty
```

v4 的資料流：

```text
OHLCV features
    ↓
Particle Filter
    ↓
hidden state estimate + uncertainty
    ↓
Long-term / Short-term Controller
    ↓
Risk Supervisor
```

## 16.2 v4 可做的東西

```text
v4.1：particle filter 估 hidden trend
v4.2：加入 uncertainty-aware exposure control
v4.3：加入 regime probability
v4.4：用 uncertainty 控制槓桿
v4.5：dynamic controller gain
```

## 16.3 v4 不應太早做的原因

如果 v3 架構還沒穩定就加入 particle filter，會發生：

```text
1. 不知道績效改善來自架構，還是 estimator。
2. Debug 困難。
3. 參數變多，overfitting 風險增加。
4. 控制器與 estimator 的責任邊界不清楚。
```

因此：

```text
v3 = 架構穩定
v4 = 數學估計器升級
```

---

# 17. v3 驗收標準

v3 不是只要報酬變高就算成功。

v3 成功條件：

```text
1. 架構清楚分成 estimator / long controller / short controller / risk supervisor / execution。
2. 每層輸出都能記錄與診斷。
3. 倉位不再只是 0/1，而是可離散控制 exposure。
4. 長線確實主導方向，短線只輔助加減碼。
5. Risk Supervisor 可以覆蓋前面訊號。
6. 回測明確扣除交易成本。
7. 沒有 look-ahead bias。
8. BTC 驗證至少不輸 v2 candidate 太多，最好改善 drawdown 或 Sharpe。
9. ETH 若仍不佳，要明確標示不是通用策略，而不是硬推。
10. 多 fee_rate 下仍有合理表現。
11. rolling validation 不能只有單一區間好看。
12. 交易次數與 turnover 對小本金可接受。
```

---

# 18. v3 與短線主導研究分支

目前主架構已決定：

```text
長線為主，短線為輔
```

反過來的架構：

```text
短線為主，長線為輔
```

暫時不作為主線，而是後續研究項目。

原因：

```text
1. 小本金下，短線交易成本影響大。
2. 短線雜訊高，假訊號多。
3. controller 容易過度反應。
4. turnover 高，可能壓低 Sharpe。
5. 若短線錯誤訊號搭配槓桿，回撤會快速放大。
6. 短線策略更容易 overfit。
```

未來研究分支名稱：

```text
Short-term Primary Experimental System
```

必備保護：

```text
long-term regime filter
daily loss limit
weekly loss limit
no-trade zone
strict transaction cost model
position cap
out-of-sample validation
```

---

# 19. v3 最終設計摘要

v3 的最終方向：

```text
v3 = 長線主導 + 短線輔助 + 風控最高權限 + 離散倉位 + 低換手率 + fee-aware execution
```

v3 要從 v2 帶過來的東西：

```text
1. regime gate 思想
2. sideways-hold-only 思想
3. weak_bull loss cooldown
4. fee-aware backtest
5. 多 fee_rate 驗證
6. BTC / ETH 分開驗證
7. 不過度相信單一最佳參數
```

v3 要新增的東西：

```text
1. Long-term Controller
2. Short-term Auxiliary Controller
3. Risk Supervisor
4. Position Composer
5. 離散 exposure 控制
6. drawdown-based risk cap
7. volatility-based risk cap
8. no-trade zone
9. 模組化 diagnostics
10. v4 estimator 替換接口
```

v3 暫時不做的東西：

```text
1. particle filter
2. continuous position
3. 高槓桿
4. 短線主導
5. 純 mean-reversion
6. 複雜最佳化器
7. 沒有經過驗證的跨資產共用參數
```

---

# 20. 一句話版本

> **v1 建立控制式交易骨架，v2 驗證 BTC 1h 風控 wrapper 有效，v3 要把這些成果重構成長線主導、短線輔助、風控最高權限的分層倉位控制系統；v4 再把 market estimation 數學化。**
