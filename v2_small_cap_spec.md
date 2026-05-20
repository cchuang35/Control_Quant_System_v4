# v2.0-small-cap 實作規格文件

> 專案：以控制看量化系統  
> 目的：將 `v1.final` 升級為適合小本金、低交易頻率、手續費敏感的 v2 系統。  
> 給 Codex 的任務重點：不要重寫 `v1.final`，而是在外層新增 regime detector、trade gate、fee-aware backtest 與診斷報表。

---

## 0. 最終設計結論

我們原本討論的 v2 是：

```text
v1.final
+ Regime Detector
+ Regime-Based Exposure Controller
+ Drawdown Brake
+ Position Smoothing
```

但考慮到目前本金不大，頻繁做連續倉位調整會被手續費吃掉，因此修正為：

```text
v2.0-small-cap.initial =
v1.final
+ Regime Detector
+ Regime-Based Trade Gate
+ Discrete Position
+ Fee-Aware Backtest
+ Validation Diagnostics
```

核心觀念：

```text
倉位控制 ≠ 頻繁交易
```

小本金版本不追求：

```text
今天 0.32 倉位
明天 0.35 倉位
後天 0.29 倉位
```

而是追求：

```text
現在允不允許進場？
現在該不該退場？
現在是不是應該完全不要交易？
```

所以 v2 的角色從「精細 exposure controller」改成「交易允許控制器 trade gate」。

---

## 1. 設計原則

### 1.1 不修改 v1.final 核心

`v1.final` 是穩定基準策略，v2 應該外掛在它外面。

```text
v1.final signal / exposure
        ↓
Regime Detector
        ↓
Trade Gate
        ↓
Final Position
```

Codex 實作時應保留 v1.final 的原始邏輯，新增 v2 wrapper，而不是直接覆寫 v1.final。

---

### 1.2 小本金優先考慮 net return

小本金下，gross return 參考價值不足，必須扣除手續費後比較。

策略淨報酬建議使用：

```text
strategy_return_net[t]
=
position[t-1] * asset_return[t]
-
abs(position[t] - position[t-1]) * fee_rate
```

其中：

```text
asset_return[t] = close[t] / close[t-1] - 1
```

注意：position 必須 lag 一天，避免 look-ahead bias。

---

### 1.3 使用離散倉位，不使用連續倉位

v2.0-small-cap 初版建議使用：

```text
position ∈ {0, 1}
```

也就是：

```text
0 = 空手
1 = 滿倉
```

暫時不使用：

```text
0.13, 0.27, 0.41, 0.58...
```

理由：

1. 降低小額調倉。
2. 降低手續費拖累。
3. 容易解釋與實盤執行。
4. 適合先驗證策略邏輯。

---

## 2. v2.0-small-cap 系統架構

完整流程：

```text
close price
   ↓
calculate MA20, MA60, momentum_20, vol20, vol60
   ↓
raw_regime
   ↓
3-day confirmation
   ↓
confirmed_regime
   ↓
v1.final signal / v1_position
   ↓
trade gate
   ↓
final_position ∈ {0, 1}
   ↓
fee-aware backtest
   ↓
metrics and diagnostics
```

模組分工：

| 模組 | 功能 |
|---|---|
| v1.final | 產生原始交易訊號或原始 exposure |
| Regime Detector | 判斷市場狀態 |
| Trade Gate | 決定是否允許 v1.final 的訊號執行 |
| Discrete Position | 將最終倉位限制在 0 / 1 |
| Fee-Aware Backtest | 扣手續費後計算績效 |
| Diagnostics | 檢查 regime 分布、交易次數、turnover、net metrics |

---

## 3. Step 1：Regime Detector

### 3.1 市場狀態分類

市場分成四種：

| Regime | 意義 | 策略態度 |
|---|---|---|
| `strong_bull` | 強多頭 | 允許進場 |
| `weak_bull` | 弱多頭 | 允許進場，但較保守 |
| `sideways` | 震盪盤 | 禁止新進場，可選擇保留既有倉位 |
| `bear` | 空頭 | 禁止進場，強制空手 |

---

### 3.2 指標定義

使用以下指標：

```text
MA20 = close 的 20 日移動平均
MA60 = close 的 60 日移動平均
momentum_20 = close / close.shift(20) - 1
vol20 = rolling_std(daily_return, 20) * sqrt(365)
vol60 = rolling_std(daily_return, 60) * sqrt(365)
high_vol = vol20 > 1.5 * vol60
```

若標的是股票日線，可將 `sqrt(365)` 改成 `sqrt(252)`。若目前主要測 Bitcoin / crypto，使用 `sqrt(365)`。

---

### 3.3 Raw Regime 規則

```python
def detect_raw_regime(price, ma20, ma60, momentum_20, vol20, vol60):
    high_vol = vol20 > 1.5 * vol60

    if price > ma60 and ma20 > ma60 and momentum_20 > 0.05 and not high_vol:
        return "strong_bull"

    elif price > ma60 and momentum_20 > 0:
        return "weak_bull"

    elif price < ma60 and ma20 < ma60:
        return "bear"

    else:
        return "sideways"
```

---

### 3.4 Regime Confirmation

為了避免 regime 每天跳動，使用 3-day confirmation。

規則：

```text
只有當 raw_regime 連續 3 天相同，confirmed_regime 才切換。
否則沿用昨天的 confirmed_regime。
```

參數：

```text
confirmation_days = 3
```

---

### 3.5 Regime Detector 輸出

需要輸出以下欄位：

```text
raw_regime
confirmed_regime
regime_score
```

建議 regime_score：

| Regime | regime_score |
|---|---:|
| strong_bull | 1.0 |
| weak_bull | 0.5 |
| sideways | 0.0 |
| bear | -1.0 |

---

## 4. Step 2：Trade Gate，而不是 Exposure Multiplier

### 4.1 原本設計被修正的地方

原本 Step 2 設計是：

```text
v2_exposure = clip(v1_exposure * regime_multiplier, 0, max_exposure)
```

但小本金下不建議這樣做，因為會產生太多小幅調倉。

修正後採用：

```text
final_position = trade_gate(v1_position, confirmed_regime, previous_position)
```

---

### 4.2 v1.final 輸出處理

如果 v1.final 已經輸出 0 / 1 訊號，直接使用：

```text
v1_position = v1.final position
```

如果 v1.final 輸出連續 exposure，先轉成 binary：

```python
v1_position = 1 if v1_exposure >= 0.5 else 0
```

參數：

```text
v1_entry_threshold = 0.5
```

---

### 4.3 Trade Gate 初版規則

建議初版採用 `sideways_hold` 版本：

| Regime | 新進場 | 既有倉位 | 最終行為 |
|---|---|---|---|
| strong_bull | 允許 | 允許 | 跟隨 v1_position |
| weak_bull | 允許 | 允許 | 跟隨 v1_position |
| sideways | 禁止 | 若 v1 仍看多，可保留 | 不新增倉位，降低震盪盤交易 |
| bear | 禁止 | 不保留 | 強制 position = 0 |

正式邏輯：

```python
def apply_trade_gate(v1_position, confirmed_regime, previous_position):
    if confirmed_regime in ["strong_bull", "weak_bull"]:
        final_position = v1_position

    elif confirmed_regime == "sideways":
        # 禁止新進場，但如果昨天已持倉且 v1 仍然看多，就保留
        if previous_position == 1 and v1_position == 1:
            final_position = 1
        else:
            final_position = 0

    elif confirmed_regime == "bear":
        final_position = 0

    else:
        final_position = 0

    return final_position
```

---

### 4.4 可選比較版本

為了知道 sideways hold 是否有幫助，可以做一個 ablation：

```text
v2.0-small-cap.strict_sideways
```

規則：

```python
if confirmed_regime in ["strong_bull", "weak_bull"]:
    final_position = v1_position
else:
    final_position = 0
```

比較：

```text
v2.0-small-cap.initial
vs
v2.0-small-cap.strict_sideways
```

---

## 5. Step 3：Drawdown Brake 的小本金修正版

### 5.1 不使用連續 drawdown multiplier

原本設計：

```text
target_exposure = regime_adjusted_exposure * drawdown_multiplier
```

小本金版本暫時不採用，因為它會導致連續倉位變化。

---

### 5.2 改成 Drawdown Risk Gate

小本金版本可以改用風險閘門：

| 策略自身 Drawdown | 交易限制 |
|---|---|
| DD > -10% | 正常交易 |
| -15% < DD <= -10% | 只允許 strong_bull 進場 |
| DD <= -15% | risk-off，強制空手 |

正式邏輯：

```python
def apply_drawdown_risk_gate(position_after_regime_gate, confirmed_regime, strategy_drawdown):
    if strategy_drawdown > -0.10:
        return position_after_regime_gate

    elif strategy_drawdown > -0.15:
        # 中度回撤：只允許 strong_bull
        if confirmed_regime == "strong_bull":
            return position_after_regime_gate
        else:
            return 0

    else:
        # 嚴重回撤：risk-off
        return 0
```

這個模組可以先做成可開關參數：

```text
use_drawdown_gate = True / False
```

第一輪必須比較：

```text
v2.0-small-cap.initial
v2.0-small-cap.no_dd_gate
```

---

## 6. Step 4：不做 Position Smoothing，改用低交易頻率規則

### 6.1 為什麼移除 smoothing

原本 smoothing 是為了處理：

```text
0.25 → 1.00
```

這種連續倉位跳變。

但現在 position 只有：

```text
0 / 1
```

所以 smoothing 不再是主角。

---

### 6.2 替代方案

低交易頻率由以下機制處理：

```text
1. regime confirmation_days = 3
2. sideways 禁止新進場
3. bear 強制空手
4. fee-aware backtest 檢查交易次數是否過多
```

可選參數：

```text
minimum_holding_days = 0 initially
cooldown_days = 0 initially
```

初版先不要加入 minimum holding 或 cooldown，避免過度複雜。若回測發現交易次數仍過多，再加入。

---

## 7. Step 5：Fee-Aware Backtest

### 7.1 必須避免 look-ahead bias

使用：

```text
strategy_return[t] = position[t-1] * asset_return[t]
```

不能使用：

```text
strategy_return[t] = position[t] * asset_return[t]
```

---

### 7.2 手續費計算

每次倉位變化都要扣手續費：

```python
trade_size = abs(position[t] - position[t-1])
fee_cost = trade_size * fee_rate
strategy_return_net[t] = position[t-1] * asset_return[t] - fee_cost
```

初始參數：

```text
fee_rate = 0.001
```

也就是單邊 0.1%。若實際交易所費率不同，Codex 應讓 fee_rate 可調。

---

### 7.3 需要輸出的回測欄位

DataFrame 至少包含：

```text
close
asset_return
v1_exposure or v1_signal
v1_position
MA20
MA60
momentum_20
vol20
vol60
raw_regime
confirmed_regime
regime_score
position_before_dd_gate
final_position
trade_size
fee_cost
strategy_return_gross
strategy_return_net
equity_net
drawdown
```

---

## 8. 驗收標準

### 8.1 主要比較版本

第一輪不要測太多，先比較：

| Version | 說明 |
|---|---|
| v1.final | 原始基準 |
| v2.0-small-cap.initial | regime gate + sideways hold + dd gate |
| v2.0-small-cap.no_dd_gate | 移除 drawdown gate |
| v2.0-small-cap.strict_sideways | sideways 也強制空手 |

---

### 8.2 主要績效指標

必須同時看 gross 與 net，但最終判斷以 net 為主。

| 類別 | 指標 |
|---|---|
| 報酬 | total return net、annualized return net |
| 風險 | max drawdown、Calmar ratio |
| 報酬品質 | Sharpe ratio、Sortino ratio |
| 交易成本 | total fee paid、turnover、number of trades |
| 交易行為 | average exposure、average holding days |
| 穩定性 | 多區間排名、regime 內表現 |

---

### 8.3 多區間驗證

至少輸出：

```text
Full period
Recent 180d
Recent 365d
Recent 2y
Recent 3y
```

每個區間都比較：

```text
annualized return net
max drawdown
Sharpe net
average exposure
turnover
number of trades
```

---

### 8.4 通過條件

v2.0-small-cap.initial 通過條件：

```text
1. net annualized return >= v1.final，或至少非常接近
2. net Sharpe >= v1.final * 0.95
3. max drawdown <= v1.final * 1.10
4. number of trades 不明顯高於 v1.final
5. turnover 不明顯高於 v1.final
6. bear regime average exposure 接近 0
7. sideways regime 不應產生大量新交易
8. strong_bull / weak_bull 才是主要持倉區間
```

若 v2 報酬提升但交易次數暴增，不能直接視為成功，必須看扣手續費後是否仍然改善。

---

## 9. Diagnostics 報表

### 9.1 總體績效表

| Version | Total Return Net | Ann. Return Net | Max DD | Sharpe Net | Sortino Net | Calmar | Avg Exposure | Trades | Turnover | Fees |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1.final |  |  |  |  |  |  |  |  |  |  |
| v2.0-small-cap.initial |  |  |  |  |  |  |  |  |  |  |
| v2.0-small-cap.no_dd_gate |  |  |  |  |  |  |  |  |  |  |
| v2.0-small-cap.strict_sideways |  |  |  |  |  |  |  |  |  |  |

---

### 9.2 Regime 分布表

| Regime | Days | Ratio |
|---|---:|---:|
| strong_bull |  |  |
| weak_bull |  |  |
| sideways |  |  |
| bear |  |  |

---

### 9.3 Regime 內表現

| Regime | Avg Position | Strategy Return Net | Asset Return | Trades | Fees |
|---|---:|---:|---:|---:|---:|
| strong_bull |  |  |  |  |  |
| weak_bull |  |  |  |  |  |
| sideways |  |  |  |  |  |
| bear |  |  |  |  |  |

---

### 9.4 交易行為表

| Metric | Value |
|---|---:|
| number_of_entries |  |
| number_of_exits |  |
| total_trades |  |
| average_holding_days |  |
| average_exposure |  |
| turnover |  |
| total_fee_paid |  |

---

## 10. Codex 實作任務 Prompts

以下 prompts 可以一步一步丟給 Codex。

---

### Prompt 1：檢查 v1.final 並建立 v2 wrapper

```text
請先閱讀目前專案中的 v1.final 策略實作，不要修改 v1.final 的核心邏輯。

目標：建立一個新的 v2.0-small-cap wrapper，使它可以讀取 v1.final 的輸出，並在外層套用 regime detector、trade gate、drawdown risk gate 與 fee-aware backtest。

請完成：
1. 找出 v1.final 目前輸出的欄位，例如 signal、position、exposure 或 strategy_return。
2. 如果 v1.final 輸出的是連續 exposure，新增參數 v1_entry_threshold=0.5，將它轉成 binary v1_position。
3. 建立新的檔案或模組，例如 v2_small_cap.py，不要覆寫 v1.final。
4. 建立清楚的函式接口：
   - compute_regime_features(df)
   - detect_raw_regime(row)
   - confirm_regime(raw_regime_series, confirmation_days=3)
   - apply_trade_gate(v1_position, confirmed_regime, previous_position)
   - apply_drawdown_risk_gate(position, confirmed_regime, strategy_drawdown)
   - backtest_v2_small_cap(df, fee_rate=0.001, use_drawdown_gate=True)

請先只建立架構與必要欄位，不要調參。
```

---

### Prompt 2：實作 Regime Detector

```text
請在 v2_small_cap.py 中實作 Regime Detector。

使用 close price 計算：
- MA20 = close rolling mean 20
- MA60 = close rolling mean 60
- momentum_20 = close / close.shift(20) - 1
- daily_return = close.pct_change()
- vol20 = daily_return.rolling(20).std() * sqrt(365)
- vol60 = daily_return.rolling(60).std() * sqrt(365)

Raw regime 規則：
- strong_bull:
  close > MA60 且 MA20 > MA60 且 momentum_20 > 0.05 且 vol20 <= 1.5 * vol60
- weak_bull:
  close > MA60 且 momentum_20 > 0
- bear:
  close < MA60 且 MA20 < MA60
- sideways:
  其他所有情況

再實作 confirmation_days=3：
只有 raw_regime 連續 3 天相同時，confirmed_regime 才切換；否則沿用前一天 confirmed_regime。

請輸出欄位：
- raw_regime
- confirmed_regime
- regime_score

regime_score 對應：
- strong_bull: 1.0
- weak_bull: 0.5
- sideways: 0.0
- bear: -1.0

請注意避免 look-ahead bias，不要使用未來資料。
```

---

### Prompt 3：實作 Trade Gate

```text
請實作 v2.0-small-cap 的 trade gate。

輸入：
- v1_position，必須是 0 或 1
- confirmed_regime
- previous_position

規則採用 sideways_hold 版本：
1. 若 confirmed_regime 是 strong_bull 或 weak_bull：
   final_position = v1_position
2. 若 confirmed_regime 是 sideways：
   禁止新進場，但如果 previous_position == 1 且 v1_position == 1，則保留倉位。
   否則 final_position = 0。
3. 若 confirmed_regime 是 bear：
   final_position = 0。
4. 其他未知狀態：
   final_position = 0。

請同時保留一個可選版本 strict_sideways：
- 只有 strong_bull / weak_bull 允許跟隨 v1_position
- sideways / bear 都強制 final_position = 0

請讓 backtest_v2_small_cap 可以用參數選擇 gate_mode：
- gate_mode="sideways_hold"
- gate_mode="strict_sideways"
```

---

### Prompt 4：實作 Drawdown Risk Gate

```text
請實作小本金版本的 drawdown risk gate，不要使用連續 exposure multiplier。

使用 v2 策略自己的 equity curve 計算 drawdown：
- equity_peak = equity.cummax()
- strategy_drawdown = equity / equity_peak - 1

Drawdown risk gate 規則：
1. 若 strategy_drawdown > -0.10：
   正常使用 regime trade gate 後的 position。
2. 若 -0.15 < strategy_drawdown <= -0.10：
   只允許 confirmed_regime == strong_bull 的交易，其他 regime position = 0。
3. 若 strategy_drawdown <= -0.15：
   risk-off，position = 0。

請讓此功能可開關：
- use_drawdown_gate=True
- use_drawdown_gate=False

注意：drawdown 必須基於 v2 自己截至當下的 equity，不可以用未來 equity。
```

---

### Prompt 5：實作 Fee-Aware Backtest

```text
請實作 fee-aware backtest，並嚴格避免 look-ahead bias。

資料欄位：
- close
- asset_return = close.pct_change()
- final_position

回測規則：
- strategy_return_gross[t] = final_position[t-1] * asset_return[t]
- trade_size[t] = abs(final_position[t] - final_position[t-1])
- fee_cost[t] = trade_size[t] * fee_rate
- strategy_return_net[t] = strategy_return_gross[t] - fee_cost[t]
- equity_net = (1 + strategy_return_net).cumprod()

參數：
- fee_rate=0.001 預設值，代表單邊 0.1%

請輸出 DataFrame，至少包含：
- close
- asset_return
- v1_position
- MA20
- MA60
- momentum_20
- vol20
- vol60
- raw_regime
- confirmed_regime
- regime_score
- position_before_dd_gate
- final_position
- trade_size
- fee_cost
- strategy_return_gross
- strategy_return_net
- equity_net
- drawdown

請確認第一筆資料與 rolling window 早期 NaN 能被安全處理。
```

---

### Prompt 6：實作績效指標與多區間比較

```text
請新增績效統計函式，用於比較 v1.final 與 v2.0-small-cap。

請計算：
- total_return_net
- annualized_return_net
- max_drawdown
- Sharpe_net
- Sortino_net
- Calmar
- average_exposure
- turnover
- number_of_entries
- number_of_exits
- total_trades
- average_holding_days
- total_fee_paid

請支援多區間比較：
- Full period
- Recent 180d
- Recent 365d
- Recent 2y
- Recent 3y

請輸出 summary table，列包含：
- version
- window
- total_return_net
- annualized_return_net
- max_drawdown
- Sharpe_net
- Sortino_net
- Calmar
- average_exposure
- turnover
- total_trades
- total_fee_paid

請注意 crypto 使用 annualization_factor=365；股票可改為 252。
```

---

### Prompt 7：實作 Regime Diagnostics

```text
請新增 regime diagnostics 報表。

請輸出：
1. Regime distribution：
   - regime
   - days
   - ratio

2. Regime performance：
   - regime
   - avg_position
   - strategy_return_net
   - asset_return
   - trades
   - fees

3. Trade behavior：
   - number_of_entries
   - number_of_exits
   - total_trades
   - average_holding_days
   - average_exposure
   - turnover
   - total_fee_paid

目的：檢查 v2 是否真的做到：
- bear regime 幾乎空手
- sideways 不產生大量新交易
- strong_bull / weak_bull 才是主要持倉區間
- 手續費沒有吃掉大部分績效
```

---

### Prompt 8：跑第一輪 ablation test

```text
請跑第一輪 ablation test，比較以下版本：

1. v1.final
2. v2.0-small-cap.initial
   - gate_mode="sideways_hold"
   - use_drawdown_gate=True
3. v2.0-small-cap.no_dd_gate
   - gate_mode="sideways_hold"
   - use_drawdown_gate=False
4. v2.0-small-cap.strict_sideways
   - gate_mode="strict_sideways"
   - use_drawdown_gate=True

請輸出：
- 總體績效比較表
- 多區間績效比較表
- regime distribution
- regime performance
- trade behavior

請不要先調參。先確認哪個模組對績效、回撤、交易次數與手續費影響最大。
```

---

### Prompt 9：檢查是否通過驗收標準

```text
請根據以下條件判斷 v2.0-small-cap.initial 是否通過：

1. net annualized return >= v1.final，或至少非常接近。
2. net Sharpe >= v1.final * 0.95。
3. max drawdown <= v1.final * 1.10。
4. total_trades 不明顯高於 v1.final。
5. turnover 不明顯高於 v1.final。
6. bear regime average exposure 接近 0。
7. sideways regime 不應產生大量新交易。
8. strong_bull / weak_bull 應該是主要持倉區間。
9. 扣除手續費後，v2 仍然有存在價值。

請輸出：
- PASS / CONDITIONAL PASS / FAIL
- 每個條件的通過狀態
- 如果 FAIL，指出最可能的問題來源：
  a. regime detector 太保守
  b. regime detector 太敏感
  c. sideways hold 造成太多虧損
  d. drawdown gate 太嚴格
  e. 手續費吃掉績效
  f. v1.final 本身訊號不足

請不要直接大幅調參，只提出下一輪最小修改建議。
```

---

## 11. 第一輪結果的解讀方式

### 11.1 如果 v2 報酬提高但交易次數也提高

必須看 net return。如果 gross 變好但 net 變差，代表交易成本吃掉優勢。

優先修正：

```text
增加 confirmation_days
使用 strict_sideways
加入 minimum_holding_days
降低 v1 進出場頻率
```

---

### 11.2 如果 v2 回撤下降但報酬也大幅下降

可能太保守。

優先檢查：

```text
strong_bull 天數比例是否太低
weak_bull 是否被 drawdown gate 擋太多
sideways 是否過度禁止交易
```

---

### 11.3 如果 v2 和 v1 差不多

代表 regime gate 沒有明顯幫助，但也不一定失敗。

下一步可以比較：

```text
no_dd_gate
strict_sideways
不同 confirmation_days
```

---

### 11.4 如果 v2 明顯變差

不要直接調參堆功能，先看 diagnostics：

```text
1. regime 分布是否合理？
2. bear 是否真的空手？
3. sideways 是否造成大量交易？
4. fee 是否過高？
5. v1_position 是否轉換錯誤？
6. 是否有 look-ahead bias 或 position lag 錯誤？
```

---

## 12. 最終摘要

v2.0-small-cap 的核心不是精細調倉，而是：

```text
用 regime detector 判斷市場環境，
用 trade gate 決定 v1.final 的訊號能不能執行，
用 fee-aware backtest 確認扣除手續費後是否真的有效。
```

最重要的設計修正：

| 原本 v2 設計 | 小本金修正版 |
|---|---|
| 連續 exposure | 0 / 1 離散倉位 |
| multiplier 調整倉位 | trade gate 決定是否允許交易 |
| drawdown multiplier | drawdown risk gate |
| position smoothing | regime confirmation + 低交易頻率規則 |
| 看 gross return | 必須看 net return after fee |

第一輪實作目標：

```text
不要追求最佳參數。
先確認 v2.0-small-cap.initial 是否在扣除手續費後，
比 v1.final 更穩、更少在錯誤 regime 交易，
且沒有讓交易次數失控。
```
