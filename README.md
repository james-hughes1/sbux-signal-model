# Starbucks Signal Model

This repo helps to construct a pipeline to create a model that predicts buy/sell signals for Starbucks stock (Nasdaq ticker: SBUX).

This comes alongside a separate [repo](https://github.com/james-hughes1/equity-dashboard) which creates a dashboard for the results ([try it here](https://sbux-model.vercel.app/)).

## Theoretical Info

The model predicts short-term alpha for SBUX, specifically the forward one-week residual alpha relative to a benchmark (SPY).

### Rolling Beta

Beta measures the sensitivity of the asset’s returns to movements in a benchmark index. We compute a time-varying (rolling) beta over a 52-week window as follows:

$$
\beta_t = \frac{\text{Cov}(r_{SBUX, t}, r_{SPY, t})}{\text{Var}(r_{SPY, t})}
$$

where:

- $ r_{SBUX, t} = $ weekly return of SBUX at time $ t $  
- $ r_{SPY, t} = $ weekly return of SPY at time $ t $  
- Cov and Var are calculated over a rolling 52-week window.  

This allows the model to capture changing market exposures over time rather than assuming a static beta.

### Residual Alpha

Residual alpha represents the portion of SBUX returns not explained by market movements, i.e., idiosyncratic performance:

$$
\alpha_t = r_{SBUX, t} - \beta_t \, r_{SPY, t}
$$

- $ \alpha_t $ is the residual (idiosyncratic) alpha at time $ t $  
- This is the main target for prediction: the forward one-week alpha:

$$
\alpha_{t+1}^{\text{fwd}} = \text{target for the model}
$$

By predicting $ \alpha_{t+1}^{\text{fwd}} $, the model provides signals that indicate whether SBUX is likely to outperform or underperform relative to its expected market exposure over the next week.

### Feature Categories

The model uses a combination of:

1. Lagged alpha features: capturing momentum and mean-reversion in idiosyncratic returns.  
2. Market and macro features: e.g., SPY returns, treasury yields, CPI, VIX, Fed funds rate changes.  
3. Equity momentum and sector signals: relative movements of SBUX, its sector (XLY), and peer stocks (MCD).  
4. Microstructure / liquidity features: e.g., volume momentum, high-low ranges, price impact, volatility.  
5. Alternative data / sentiment signals: e.g., Google Trends interest in Starbucks.

These features aim to capture both macro-driven risk factors and short-term idiosyncratic opportunities in the stock.

### Model Objective

By combining these signals in a supervised regression framework (currently Ridge regression), the model estimates expected forward alpha:

$$
\hat{\alpha}_{t+1}^{\text{fwd}} = f(\text{lagged alpha, market factors, microstructure, sentiment, …})
$$

This predicted alpha can be used as a quantitative signal to inform trading or portfolio allocation decisions for SBUX.


## How to use

Set up environment using `pip install -r requirements.txt`

Configure the pipeline using the jsons in `src/config/`

Then run sequentially the scripts in `src/` e.g. `python run 01_collect.py`

A lot of data can be produced in various runs and reruns of the pipeline stages, it can be cleaned up safely using `python src/clean.py`; if you just want to target particular stages you can add options based on the directory names, such as `--model`.

## Acknowledgements

This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.