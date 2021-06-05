import argparse
import os
import sys
import webbrowser
from datetime import timedelta

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import requests_cache
from plotly import graph_objs as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from finance_benchmark import config


def get_asset_data(assets_ticker, startdate):
    """Retrieve assets data from yahoo finance

    Args:
        assets_ticker ([str]): list of assets to download
        startdate (str): start date
    """
    df = pd.DataFrame()
    for ticker in tqdm(assets_ticker):  # progress bar when downloading data
        ticker = ticker.strip()
        # cache data to avoid downloading them again when filtering
        session = requests_cache.CachedSession(
            cache_name="../cache", backend="sqlite", expire_after=timedelta(days=1)
        )
        try:
            # Get daily closing price
            data = web.DataReader(
                ticker, data_source="yahoo", start=startdate, session=session
            )["Close"]
        except Exception:
            print("Error fetching : " + ticker)
            continue

        data = pd.DataFrame({"Date": data.index, ticker: data.values})
        data.drop_duplicates(
            subset="Date", inplace=True
        )  # remove duplicate statement for the same day
        data.set_index("Date", inplace=True)
        df = df.join(data, how="outer")  # add asset data to main dataframe

    df.sort_index(inplace=True)
    df.dropna(axis=1, how="all", inplace=True)  # remove null values

    # remove assets with less than 60 days of data history
    for col in df.columns:
        if (len(df) - df[col].isna().sum()) < 60:
            df.drop(col, axis=1, inplace=True)

    df.drop_duplicates(inplace=True)

    df.to_csv("../assets.csv")


# For later improvement it may be better to use sortino ratio instead of sharpe ratio
# https://www.investopedia.com/ask/answers/010815/what-difference-between-sharpe-ratio-and-sortino-ratio.asp
def optimize_sharpe_ratio(num_portfolios):
    """Optimize portfolio with sharpe ratio

    Generate `num_portfolios` portfolio with different proportions for each given asset.
    It will calculate sharpe ratio of each one of this portfolio to try to find the best performing one with volatily as low as possible.

    Args:
        num_portfolios (int): number of portfolio to generate

    Returns:
        [type]: [description]
    """
    data = pd.read_csv("../assets.csv")
    data.set_index("Date", inplace=True)
    if "Portfolio" in data.columns:
        data.drop("Portfolio", 1, inplace=True)

    asset_size = len(data.columns)
    # convert daily asset prices into daily returns
    returns = data.pct_change()
    # calculate mean daily return and covariance of daily returns
    mean_daily_returns = np.array(returns.mean())
    cov_matrix = np.array(returns.cov())
    # set up array to hold results
    # Increase the size of the array to hold the return, std deviation and sharpe ratio
    results = np.zeros((asset_size + 3, num_portfolios))
    results = results.tolist()
    for i in range(num_portfolios):
        # select random weights for portfolio holdings
        weights = np.array(np.random.random(asset_size))
        # rebalance weights to sum to 1
        weights /= np.sum(weights)

        # calculate portfolio return and volatility
        portfolio_return = np.sum(mean_daily_returns * weights) * 252
        portfolio_std_dev = np.sqrt(
            np.dot(weights.T, np.dot(cov_matrix, weights))
        ) * np.sqrt(252)

        # store returns and standard deviation in results array
        # store Sharpe Ratio (return / volatility) - risk free rate element excluded for simplicity
        results[0][i] = portfolio_return
        results[1][i] = portfolio_std_dev
        results[2][i] = results[0][i] / results[1][i]
        # iterate through the weight vector and add data to results array
        weights = weights.tolist()
        for j, weight in enumerate(weights):
            results[j + 3][i] = weight
    # convert results array to Pandas DataFrame
    my_columns = ["returns", "stdev", "sharpe"]
    my_columns.extend(data.columns)

    results = np.array(results)
    results_frame = pd.DataFrame(results.T, columns=my_columns)
    # Portfolio with highest Sharpe Ratio
    max_sharpe_portfolio = results_frame.iloc[results_frame["sharpe"].idxmax()]
    # Portfolio with minimum standard deviation
    min_vol_port = results_frame.iloc[results_frame["stdev"].idxmin()]

    # create scatter plot coloured by Sharpe Ratio
    fig = go.Figure(
        data=(
            go.Scattergl(
                x=results_frame.stdev,
                y=results_frame.returns,
                mode="markers",
                marker=dict(
                    color=results_frame.sharpe,
                    colorbar=dict(title="Sharpe ratio"),
                    colorscale="bluered",
                    reversescale=True,
                ),
            )
        )
    )

    # Add asset proportion of portfolio with best sharpe ratio
    fig.add_annotation(
        text=max_sharpe_portfolio.to_string().replace("\n", "<br>"),
        align="left",
        showarrow=False,
        xref="paper",
        yref="paper",
        x=1,
        y=0,
        bordercolor="black",
        borderwidth=1,
    )
    fig.add_annotation(
        x=max_sharpe_portfolio[1], y=max_sharpe_portfolio[0], text="Best sharpe ratio"
    )
    fig.add_annotation(x=min_vol_port[1], y=min_vol_port[0], text="Lower volatility")

    fig.update_layout(xaxis_title="Volatility", yaxis_title="Returns")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.write_html("../result_html/sharpe_fig.html", auto_open=False, full_html=False)

    print("####### Optimized portfolio #######")
    print(max_sharpe_portfolio)
    print("###################################")

    return max_sharpe_portfolio


def sharpe_each_asset_chart():
    """Get the sharpe ratio of each asset

    Returns:
        list: list of assets with less than 0.2 sharpe ratio.
    """
    data = pd.read_csv("../assets.csv")
    data.set_index("Date", inplace=True)

    # convert daily asset prices into daily returns
    returns = data.pct_change()
    sharpe_ratio = (returns.mean() / returns.std()) * (252 ** 0.5)

    mean_return = returns.mean() * 252
    volatility = returns.std() * (252 ** 0.5)

    print("Sharpe ratio for each asset: ")
    ticker_to_eliminate = [
        key for key, value in sharpe_ratio.to_dict().items() if value <= 0.2
    ]
    print(dict(zip(sharpe_ratio.index.to_list(), sharpe_ratio.to_list())))
    fig = go.Figure(
        data=(
            go.Scattergl(
                x=volatility,
                y=mean_return,
                text=sharpe_ratio.index.to_list(),
                hovertext=sharpe_ratio.to_list(),
                textposition="top center",
                mode="markers+text",
                marker=dict(
                    color=sharpe_ratio,
                    colorbar=dict(title="Sharpe ratio"),
                    colorscale="bluered",
                    reversescale=True,
                ),
            )
        )
    )

    fig.add_shape(
        # diagonal line
        type="line",
        x0=0,
        y0=0,
        x1=10,
        y1=10,
        line=dict(
            color="MediumPurple",
            width=2,
        ),
    )

    fig.update_layout(
        xaxis_title="Volatility",
        yaxis_title="Returns",
        yaxis=dict(range=[0, 1]),
        xaxis=dict(range=[0, 1]),
    )

    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.write_html(
        "../result_html/sharpe_by_asset_fig.html", auto_open=False, full_html=False
    )

    return ticker_to_eliminate


def evol_chart(weight):
    """Performance chart of each asset
    also include optimized portfolio performance

    Args:
        weight ([int]): weight of each asset in portfolio
    """

    fig = go.Figure()

    df = pd.read_csv("../assets.csv")
    df.set_index("Date", inplace=True)

    df = df.loc[~df.index.duplicated(keep="first")]
    df_final = df.copy()

    for col in df.columns:
        df[col] = df[col] / df[col].at[df[col].first_valid_index()] - 1

    # Add optimized portfolio performance line in the graph
    df_portfolio = df.copy()
    df_portfolio.dropna(inplace=True)
    for i in range(len(df_portfolio.columns)):
        df_portfolio.iloc[:, [i]] = df_portfolio.iloc[:, [i]] * weight[i]
    df_portfolio["Portfolio"] = df_portfolio.sum(axis=1)
    df["Portfolio"] = df_portfolio["Portfolio"]

    polite_name = config.polite_name
    for col in df.columns:
        polite_ticker = polite_name[col] if col in polite_name else col
        data_col = df[col].dropna()

        fig.add_trace(
            go.Scattergl(
                x=data_col.index, y=data_col.values, mode="lines", name=polite_ticker
            )
        )

    fig.layout.yaxis.tickformat = ",.0%"

    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.write_html(
        "../result_html/evol_asset_price.html", auto_open=False, full_html=False
    )

    df_final["Portfolio"] = df["Portfolio"]
    df_final.to_csv("../assets.csv")


def resample_portfolio_period(weight, period):
    """Change portfolio period index by year or by month
    Resample time-series data.

    Args:
        weight ([int]): weight of each asset in portfolio
        period (str): 'Y': yearly and 'M': monthly

    Returns:
        dataframe: resampled portfolio
    """
    df = pd.read_csv("../assets.csv")

    df.drop("Portfolio", axis=1, inplace=True)
    df.set_index("Date", inplace=True)

    df.index = pd.to_datetime(df.index)
    df = df.fillna(method="ffill")

    resampled_df = df.copy()

    resampled_df = resampled_df.loc[~resampled_df.index.duplicated(keep="first")]
    resampled_df = resampled_df.resample(period, convention="s").ffill()

    resampled_df.dropna(inplace=True)
    for i in range(len(resampled_df.columns)):
        resampled_df.iloc[:, [i]] = resampled_df.iloc[:, [i]] * weight[i]

    resampled_df["Portfolio"] = resampled_df.sum(axis=1)
    resampled_df["Portfolio"] = resampled_df["Portfolio"].pct_change()

    return resampled_df


def average_period_performance(df, period):
    """Calculate average performance for a certain period

    Args:
        df (dataframe): dataframe with portfoilio performance for each period
        period (str): concerned period
    """
    try:
        return_each_period = df["Portfolio"].tolist()[1:]
        total_return = 1
        for val in return_each_period:
            total_return *= 1 + val
        average_period_return = (total_return ** (1 / len(return_each_period))) - 1
        print(
            f"Average {period} performance : {'{0:.1%}'.format(average_period_return)}"
        )
    except Exception:
        print(f"Error : average {period} return calculation")


def period_return_chart(weight):
    """Chart of performance per month and performance per year
    Might be a little buggy with some bad data need improvement
    Args:
        weight ([int]): weight of each asset in portfolio

    Returns:
        [type]: [description]
    """
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.05)

    df_annual = resample_portfolio_period(weight, "Y")
    df_annual.index = [str(date)[:4] for date in df_annual.index.to_list()]

    average_period_performance(df_annual, "annual")

    df_negative_year = df_annual[df_annual["Portfolio"] < 0]
    df_positive_year = df_annual[df_annual["Portfolio"] > 0]

    # Trace green bar for positive year return
    fig.add_trace(
        go.Bar(
            x=df_positive_year.index,
            y=df_positive_year["Portfolio"].to_list(),
            text=df_positive_year["Portfolio"].to_list(),
            textposition="auto",
            texttemplate="%{text:.0%}",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    # Trace red bar for negative year return
    fig.add_trace(
        go.Bar(
            x=df_negative_year.index,
            y=df_negative_year["Portfolio"].to_list(),
            text=df_negative_year["Portfolio"].to_list(),
            textposition="auto",
            texttemplate="%{text:.0%}",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    df_monthly = resample_portfolio_period(weight, "M")
    df_monthly.index = [str(date)[:7] for date in df_monthly.index.to_list()]
    average_period_performance(df_monthly, "month")

    df_negative_month = df_monthly[df_monthly["Portfolio"] < 0]
    df_positive_month = df_monthly[df_monthly["Portfolio"] > 0]

    # Trace green bar for positive month return
    fig.add_trace(
        go.Bar(
            x=df_positive_month.index,
            y=df_positive_month["Portfolio"].to_list(),
            text=df_positive_month["Portfolio"].to_list(),
            textposition="auto",
            texttemplate="%{text:.0%}",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    # Trace red bar for negative month return
    fig.add_trace(
        go.Bar(
            x=df_negative_month.index,
            y=df_negative_month["Portfolio"].to_list(),
            text=df_negative_month["Portfolio"].to_list(),
            textposition="auto",
            texttemplate="%{text:.0%}",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.layout.yaxis1.tickformat = ",.0%"
    fig.layout.yaxis2.tickformat = ",.0%"

    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

    fig.write_html(
        "../result_html/period_return_fig.html", auto_open=False, full_html=False
    )


def create_final_html_file():
    """Assembles all graphs together in a single HTMl file named DASHBOARD.html
    and open a page in browser
    """
    with open("../result_html/DASHBOARD.html", "w+") as html_graph_file:
        html_graph_file.write("<html><head></head><body>" + "\n")

        html_graph_file.write(
            ' <div style= "position: relative;  top: 0px;  left: 0;  width: 68%;  height: 59%;">'
            + "\n"
        )
        html_graph_file.write(
            '  <object data="'
            + "evol_asset_price.html"
            + '" width="100%" height="100%"></object></div>'
            + "\n"
        )
        html_graph_file.write(' <div style= "position: relative; ">' + "\n")
        html_graph_file.write(
            ' <object data="'
            + "period_return_fig.html"
            + '" width="62%" height="42%"></object>'
            + "\n"
        )
        html_graph_file.write(" </div>" + "\n")
        html_graph_file.write(
            ' <div style= "position: fixed;  top: 0px;  right: 0;  width: 33%;  height: 100%;">'
            + "\n"
        )
        html_graph_file.write(
            '  <object data="'
            + "sharpe_fig.html"
            + '" width="100%" height="50%"></object>'
            + "\n"
        )
        html_graph_file.write(
            '  <object data="'
            + "sharpe_by_asset_fig.html"
            + '" width="100%" height="50%"></object>'
            + "\n"
        )
        html_graph_file.write(" </div>" + "\n")

        html_graph_file.write("</body></html>")
    webbrowser.open_new_tab("../result_html/DASHBOARD.html")


def get_data_and_create_graph(assets, startdate, num_portfolio):
    """Main function which will retrieve assets data and create graphs
    based on it
    Args:
        assets (list): Assets to analyze
        startdate (string): Date where the analyze should begin
        num_portfolio (int): number of different portfolio allocation to create
    """
    get_asset_data(assets, startdate)

    max_sharpe_portfolio = optimize_sharpe_ratio(num_portfolio)
    ticker_to_eliminate = sharpe_each_asset_chart()
    evol_chart(max_sharpe_portfolio.tolist()[3:])
    period_return_chart(max_sharpe_portfolio.tolist()[3:])

    create_final_html_file()
    return max_sharpe_portfolio, ticker_to_eliminate


def main(args):
    # Check config variable are set
    for variable in ["startdate", "num_portfolio", "assets"]:
        if not hasattr(config, variable):
            print(f'Missing variable "{variable}" in config file')
            sys.exit(1)

    startdate = config.startdate
    num_portfolio = config.num_portfolio
    assets = config.assets
    # assets with percentage less than this value will be deleted in optimized portfolio
    value_filter = 100 / len(assets) * (35 / 10000)

    max_sharpe_portfolio, ticker_to_eliminate = get_data_and_create_graph(
        assets, startdate, num_portfolio
    )

    for _ in range(args.filter):
        max_sharpe_portfolio = max_sharpe_portfolio[
            max_sharpe_portfolio.values > value_filter
        ]
        print(max_sharpe_portfolio)
        print("############################")
        print("FILTRATION...")
        print("############################")

        new_assets = max_sharpe_portfolio.index.tolist()[3:]
        new_assets = [x for x in new_assets if x not in ticker_to_eliminate]
        if assets == new_assets:  # if no filtration is made this is the "final" result
            break
        assets = new_assets

        max_sharpe_portfolio, ticker_to_eliminate = get_data_and_create_graph(
            assets, startdate, num_portfolio
        )


if __name__ == "__main__":
    os.chdir(os.path.dirname(sys.argv[0]))
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", dest="filter", type=int, help="filtered mode", default=0)
    main(parser.parse_args())
