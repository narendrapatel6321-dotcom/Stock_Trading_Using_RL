# Deep Reinforcement Learning for Stock Trading (NIFTY 50)

This repository implements a reinforcement learning–based stock trading agent using **Proximal Policy Optimization (PPO)** for the Indian equity market.

The project formulates portfolio trading as a Markov Decision Process (MDP) and learns a trading policy that directly optimizes risk-adjusted portfolio performance, rather than predicting future prices.

The implementation is inspired by the ICAIF-2020 paper *"Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy"*, adapted to **NIFTY 50** stocks with realistic market constraints and a rolling retraining protocol.

---

## Key Features

### Custom Trading Environment
* Portfolio trading modeled as an MDP.
* Explicit cash, holdings, prices, and technical indicators.

### Continuous Action Space
* Direct position sizing per stock (buy/sell/hold is implicitly learned).
* Actions operate within a continuous space rather than discrete signals.

### Realistic Market Constraints
* **Transaction Costs:** 0.1% per trade.
* **Liquidity:** Non-negative cash balance enforcement.

### Risk Control via Turbulence Index
* Uses Mahalanobis-distance–based market turbulence.
* Implements buy restrictions and forced liquidation during extreme market conditions.

### Rolling Window Training
* Periodic retraining on expanding historical data.
* Out-of-sample quarterly trading evaluation.

### Cross-Market Robustness Test
* PPO policy trained on Indian data is evaluated on US equities to test generalization.

---

## Methodology

### State Space
At each time step $t$, the agent observes:

$$
s_t = [b_t, P_t, h_t, MACD_t, RSI_t, CCI_t, ADX_t]
$$

Where:
* $b_t$: Available cash balance
* $P_t$: Vector of stock prices
* $h_t$: Vector of shares held
* *Technical indicators are computed per stock.*

### Action Space
* Continuous actions in $[-1, 1]$ for each stock.
* Actions are mapped to the number of shares bought or sold.
* **Automatic Clipping:** Actions are adjusted to satisfy cash constraints and maximum position limits.

### Reward Function
The reward is defined as the change in portfolio value:

$$
r_t = (b_{t+1} + P_{t+1}^\top h_{t+1}) - (b_t + P_t^\top h_t) - \text{transaction cost}
$$

This directly aligns the agent’s objective with portfolio growth under costs, rather than short-term price movements.

### Learning Algorithm: Proximal Policy Optimization (PPO)
Chosen for:
* Stability in continuous action spaces.
* Controlled policy updates.
* Robust convergence in noisy financial environments.

---

## Experimental Setup

* **Market:** NIFTY 50 (fixed constituents)
* **Data Period:** 2010 – 2019
* **Initial Capital:** ₹1,000,000
* **Training:** Expanding window (from 2010 onward)
* **Trading:** Rolling quarterly out-of-sample evaluation
* **Benchmarks:**
    * NIFTY 50 index
    * Equal-weight Buy & Hold portfolio
* **Metrics:** Cumulative return, Sharpe ratio, Maximum drawdown, Volatility

---

## Results

### Overall Performance (2016–2019)

| Strategy | Final Value | Return | Sharpe | Max Drawdown | Volatility |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PPO Agent** | **₹1,729,865** | **72.99%** | **0.61** | **−22.91%** | **15.62%** |
| NIFTY 50 | ₹1,561,800 | 56.18% | 0.50 | −14.55% | 13.00% |
| Buy & Hold (Equal Wt) | ₹1,322,604 | 32.26% | 0.16 | −24.32% | 16.21% |

> The PPO agent achieves higher risk-adjusted returns than both passive benchmarks while maintaining controlled volatility.

---

## Visual Analysis

### Portfolio vs Benchmarks
The PPO agent consistently outperforms NIFTY 50 and equal-weight Buy & Hold over the evaluation period.

![Portfolio comparison](images/plot_1.png)

### Market Events Sensitivity
Performance is analyzed around major Indian macroeconomic events:
1.  **Demonetization (2016)**
2.  **GST Rollout (2017)**
3.  **PNB–Nirav Modi Scandal (2018)**
4.  **Corporate Tax Cut (2019)**

The agent adapts dynamically, reducing exposure during periods of heightened uncertainty.

![Major Events](images/plot_2.png)

### Cross-Market Generalization
A PPO policy trained on Indian market data is evaluated on US equities, demonstrating partial generalization and reinforcing that the agent learns market-agnostic trading structure, not market-specific memorization.

![Market_comp](images/plot_3.png)

---

## Key Observations

* PPO effectively learns dynamic position sizing, not static allocation.
* Risk-adjusted performance improves during trending markets.
* Drawdowns are mitigated via turbulence-aware risk control.
* Rolling retraining helps adapt to changing market regimes.

---

## References

* Yang et al., *Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy*, ICAIF 2020
* OpenAI Gym
* Stable-Baselines PPO
