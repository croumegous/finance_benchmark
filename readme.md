## Disclaimer
I made this project a year ago, so the source code is quite bad.
I tried to clean it recently to make it public but it is not perfect yet.

:warning: This project is not totally reliable and the data shown can be buggy/biased, do your own research and don't take conclusion blindly on data shown in this project.


# Presentation
This repository is made to optimize performance of your saving portfolio.
Based on [sharpe ratio ](https://en.wikipedia.org/wiki/Sharpe_ratio) to optimize performance over volatility.


You can configure the ticker you want to benchmark in `config.py` file.
This benchmark take data from yahoo finance you can use all available [ticker](https://en.wikipedia.org/wiki/Ticker_symbol) of yahoo :  stocks, ETF, crypto ...

The source code will generate a large number of portfolio with different proportions for each given asset.  
Then it will calculate sharpe ratio of each one of this portfolio to try to find the best performing one with volatily as low as possible.  
Then it will return the best proportion to be used in the portfolio for each ticker.

### Example:
If I put `stocks = ["BTC-USD", "^IXIC", "^GSPC"]` (ticker for Bitcoin, NASDAQ-100 and S&P500 respectively)
and BTC as a great sharpe ratio compare to S&P500 and NASDAQ-100 the return value could be something like :  
"BTC-USD": 0.7
"^IXIC": 0.2
"^GSPC": 0.1

This mean that the best portfolio for the period of time given would be to have a portfolio composed of 70% BTC, 20% NASDAQ-100 and 10% S&P500.  
:warning: Be carefull not to trust the returned values blindly, it's an experimental tool made for fun and learning purpose I wouldn't personnaly based my investement on it.



It will show different graph at the end of the process, the first graph will shown performance of each asset, the second one the sharpe ratio of all the portfolio generated.  
The 3 one performance per month and per year for the "optimized" portfolio, and the last one the sharpe ratio of each asset.


(put image of a result with number legend (1), (2) etc....)


Visualization is made in HTML with Plotly.
I used pandas and numpy to handle data analysis.


# Usage
install [python](https://www.python.org/downloads/) 3.8 or above and [poetry](https://python-poetry.org/docs/) (pakage manager) on your computer


Once done use 
```bash
poetry install  
```
at the root folder to install all required dependencies.

Run
```bash
poetry run python finance_benchmark/src/portfolio_optimization.py
```

You can also use -f <int> to filter assets wich means that the algorithm will be ran multiple time and each time it will remove the less performing assets. This is usefull when you compare a lot of assets.


## Configuration 
You can configure the ticker you want to use inside `config.py` file, by using an array called `assets`.
You can also change the `startdate`, and the number of portfolio (`num_portfolio` wich will be randomly generated.  
Ultimately you can change `polite_name` variable to rename ticker wich will be shown on the graph.
