# https://www.pythonforfinance.net/2017/01/21/investment-portfolio-optimisation-with-python/

import argparse
import os
import sys
import webbrowser
from datetime import timedelta
from math import sqrt

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import requests_cache
from plotly import graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots
from tqdm import tqdm

from finance_benchmark import config


def get_asset_data(assets_ticker, startdate):
    """Retrieve assets data from yahoo finance

    Args:
        assets_ticker ([type]): [description]
        startdate ([type]): [description]
    """
    first = True
    for ticker in tqdm(assets_ticker):  # progress bar when downloading data
        ticker = ticker.strip()
        # cache data to avoid downloading them again when filtering
        session = requests_cache.CachedSession(
            cache_name="cache", backend="sqlite", expire_after=timedelta(days=3)
        )
        try:
            # Get daily closing price
            data = web.DataReader(
                ticker, data_source="yahoo", start=startdate, session=session
            )["Close"]
        except Exception:
            print("Error fetching : " + ticker)
            continue

        if first:
            df = data
            df = pd.DataFrame({"Date": df.index, ticker: df.values})
            df.drop_duplicates(subset="Date", inplace=True)
            df.set_index("Date", inplace=True)
            first = False
        else:
            data = pd.DataFrame({"Date": data.index, ticker: data.values})
            data.drop_duplicates(subset="Date", inplace=True)
            data.set_index("Date", inplace=True)
            df = df.join(data, how="outer")

    df.sort_index(inplace=True)
    df.dropna(axis=1, how="all", inplace=True)

    # remove assets with less than 60 days of data
    for col in df.columns:
        if (len(df) - df[col].isna().sum()) < 60:
            df.drop(col, axis=1, inplace=True)

    df.drop_duplicates(inplace=True)

    df.to_csv("new_assets.csv")


# For later improvement it may be better to use sortino ratio instead of sharpe ratio
# https://www.investopedia.com/ask/answers/010815/what-difference-between-sharpe-ratio-and-sortino-ratio.asp
def optimize(num_portfolios):
    """Optimize portfolio with sharpe ratio

    Args:
        num_portfolios ([type]): [description]

    Returns:
        [type]: [description]
    """
    data = pd.read_csv("new_assets.csv")
    data.set_index("Date", inplace=True)
    if "Portfolio" in data.columns:
        data.drop("Portfolio", 1, inplace=True)
    print(data.head())
    print(data.tail())

    asset_size = len(data.columns)
    # convert daily asset prices into daily returns
    returns = data.pct_change()
    # calculate mean daily return and covariance of daily returns
    mean_daily_returns = np.array(returns.mean())
    cov_matrix = np.array(returns.cov())
    # set up array to hold results
    # We have increased the size of the array to hold the weight values for each asset
    results = np.zeros((4 + asset_size - 1, num_portfolios))
    results = results.tolist()
    for i in range(num_portfolios):
        # select random weights for portfolio holdings
        weights = np.array(np.random.random(asset_size))
        # print(weights)
        # rebalance weights to sum to 1
        weights /= np.sum(weights)

        # calculate portfolio return and volatility
        portfolio_return = np.sum(mean_daily_returns * weights) * 252
        # print(np.dot(cov_matrix, weights))
        portfolio_std_dev = np.sqrt(
            np.dot(weights.T, np.dot(cov_matrix, weights))
        ) * sqrt(252)

        # store results in results array
        results[0][i] = portfolio_return
        results[1][i] = portfolio_std_dev
        # store Sharpe Ratio (return / volatility) - risk free rate element excluded for simplicity
        results[2][i] = results[0][i] / results[1][i]
        # iterate through the weight vector and add data to results array
        weights = weights.tolist()
        for j in range(len(weights)):
            results[j + 3][i] = weights[j]
        weights = np.array(weights)
    # convert results array to Pandas DataFrame
    # print(results)
    my_columns = ["ret", "stdev", "sharpe"]
    my_columns.extend(data.columns)

    results = np.array(results)
    results_frame = pd.DataFrame(results.T, columns=my_columns)
    # locate position of portfolio with highest Sharpe Ratio
    max_sharpe_port = results_frame.iloc[results_frame["sharpe"].idxmax()]
    # locate positon of portfolio with minimum standard deviation
    min_vol_port = results_frame.iloc[results_frame["stdev"].idxmin()]
    # create scatter plot coloured by Sharpe Ratio

    fig = go.Figure(
        data=(
            go.Scattergl(
                x=results_frame.stdev,
                y=results_frame.ret,
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

    fig.add_annotation(
        text=max_sharpe_port.to_string().replace("\n", "<br>"),
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
        x=max_sharpe_port[1], y=max_sharpe_port[0], text="Best sharpe ratio"
    )
    fig.add_annotation(x=min_vol_port[1], y=min_vol_port[0], text="Lower volatility")

    fig.update_layout(xaxis_title="Volatility", yaxis_title="Returns")

    # plot(fig, 'u.html', auto_open=True)

    # pd.options.display.max_colwidth = 100
    print("RESULT =============================")
    print(max_sharpe_port)
    print("====================================")

    return fig, max_sharpe_port


def sharpe_each_asset_chart():
    """Get the sharpe ratio of each asset

    Returns:
        [type]: [description]
    """
    data = pd.read_csv("new_assets.csv")
    data.set_index("Date", inplace=True)

    # convert daily asset prices into daily returns
    returns = data.pct_change()
    sharpe_ratio = (returns.mean() / returns.std()) * (252 ** 0.5)

    mean_return = returns.mean() * 252
    volatility = returns.std() * (252 ** 0.5)

    print("ALL SHARPE : ")
    ticker_to_eliminate = [
        key for key, value in sharpe_ratio.to_dict().items() if value <= 0.2
    ]
    print(sharpe_ratio.index.to_list())
    print(sharpe_ratio.to_list())
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
        # Line Diagonal
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(
            color="MediumPurple",
            width=2,
        ),
    )

    fig.update_layout(
        xaxis_title="Volatility",
        yaxis_title="Returns",
        yaxis=dict(range=[0, 0.5]),
        xaxis=dict(range=[0, 0.5]),
    )

    return fig, ticker_to_eliminate
    # plot(fig, 'u.html', auto_open=True)


def evol_chart(weight, log=False):
    """[summary]

    Args:
        weight ([type]): [description]

    Returns:
        [type]: [description]
    """

    fig = go.Figure()

    df = pd.read_csv("new_assets.csv")
    df.set_index("Date", inplace=True)

    df = df.loc[~df.index.duplicated(keep="first")]
    df_final = df.copy()

    for col in df.columns:
        df[col] = df[col] / df[col].at[df[col].first_valid_index()] - 1

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

    dropdownScaleViewer = list(
        [
            dict(
                active=1,
                x=0.11,
                xanchor="left",
                y=1.1,
                yanchor="top",
                # Add button on graph to change from log to linear
                buttons=list(
                    [
                        dict(
                            label="Log Scale",
                            method="update",
                            args=[
                                {"visible": [True, True]},
                                {"yaxis": {"type": "log"}},
                            ],
                        ),
                        dict(
                            label="Linear Scale",
                            method="update",
                            args=[
                                {"visible": [True, False]},
                                {"yaxis": {"type": "linear"}},
                            ],
                        ),
                    ]
                ),
            )
        ]
    )
    fig.layout = dict(updatemenus=dropdownScaleViewer)
    fig.layout.yaxis.tickformat = ",.0%"
    # if log:
    #     fig.layout.yaxis.type = "log"

    df_final["Portfolio"] = df["Portfolio"]
    df_final.to_csv("new_assets.csv")

    return fig


def resample_portfolio_change(weight, period):
    """[summary]

    Args:
        weight ([type]): [description]
        period ([type]): [description]

    Returns:
        [type]: [description]
    """
    df = pd.read_csv("new_assets.csv")

    df.drop("Portfolio", axis=1, inplace=True)
    df.set_index("Date", inplace=True)

    df.index = pd.to_datetime(df.index)
    df = df.fillna(method="ffill")

    df_temp = df.copy()

    df_temp = df_temp.loc[~df_temp.index.duplicated(keep="first")]
    df_temp = df_temp.resample(period, convention="s").ffill()

    df_temp.dropna(inplace=True)
    for i in range(len(df_temp.columns)):
        df_temp.iloc[:, [i]] = df_temp.iloc[:, [i]] * weight[i]

    df_temp["Portfolio"] = df_temp.sum(axis=1)
    df_temp["Portfolio"] = df_temp["Portfolio"].pct_change()

    return df_temp


def period_return_chart(weight):
    """Chart of performance per month and performance per year
    Might be a little buggy with some bad data need improvement
    Args:
        weight ([type]): [description]

    Returns:
        [type]: [description]
    """
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.05)

    df_annual = resample_portfolio_change(weight, "Y")
    df_annual.index = [str(date)[:4] for date in df_annual.index.to_list()]

    try:
        ann_return = df_annual["Portfolio"].tolist()[1:]
        annualized_ret = 1
        for val in ann_return:
            annualized_ret *= 1 + val
        annualized_ret = (annualized_ret ** (1 / len(ann_return))) - 1
        print("ANNUALIZED RETURN : " + str(annualized_ret))
    except Exception:
        print("Erreur : calcul rendement annualisé ")

    df_negative_year = df_annual[df_annual["Portfolio"] < 0]
    df_positive_year = df_annual[df_annual["Portfolio"] > 0]

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

    df_monthly = resample_portfolio_change(weight, "M")
    df_monthly.index = [str(date)[:7] for date in df_monthly.index.to_list()]

    df_negative_month = df_monthly[df_monthly["Portfolio"] < 0]
    df_positive_month = df_monthly[df_monthly["Portfolio"] > 0]

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

    return fig


def create_final_review(
    evol_price_fig, sharpe_fig, sharpe_by_asset_fig, period_return_fig
):
    """Assembles all graphs together in a single HTMl file named DASHBOARD.html
    and open a page in browser
    Args:
        evol_price_fig ([type]): [description]
        sharpe_fig ([type]): [description]
        sharpe_by_asset_fig ([type]): [description]
        period_return_fig ([type]): [description]
    """
    fichier_html_graphs = open("../result_html/DASHBOARD.html", "w+")
    fichier_html_graphs.write("<html><head></head><body>" + "\n")

    evol_price_fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        # paper_bgcolor="LightSteelBlue",
    )
    sharpe_fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        # paper_bgcolor="LightSteelBlue",
    )
    sharpe_by_asset_fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        # paper_bgcolor="LightSteelBlue",
    )
    period_return_fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        # paper_bgcolor="LightSteelBlue",
    )

    plot(
        evol_price_fig, filename="../result_html/evol_asset_price.html", auto_open=False
    )
    plot(sharpe_fig, filename="../result_html/sharpe_fig.html", auto_open=False)
    plot(
        period_return_fig,
        filename="../result_html/period_return_fig.html",
        auto_open=False,
    )
    plot(
        sharpe_by_asset_fig,
        filename="../result_html/sharpe_by_asset_fig.html",
        auto_open=False,
    )

    fichier_html_graphs.write(
        ' <div style= "position: relative;  top: 0px;  left: 0;  width: 68%;  height: 59%;">'
        + "\n"
    )
    fichier_html_graphs.write(
        '  <object data="'
        + "evol_asset_price.html"
        + '" width="100%" height="100%"></object></div>'
        + "\n"
    )
    fichier_html_graphs.write(' <div style= "position: relative; ">' + "\n")
    fichier_html_graphs.write(
        ' <object data="'
        + "period_return_fig.html"
        + '" width="62%" height="42%"></object>'
        + "\n"
    )
    fichier_html_graphs.write(" </div>" + "\n")
    fichier_html_graphs.write(
        ' <div style= "position: fixed;  top: 0px;  right: 0;  width: 33%;  height: 100%;">'
        + "\n"
    )
    fichier_html_graphs.write(
        '  <object data="'
        + "sharpe_fig.html"
        + '" width="100%" height="50%"></object>'
        + "\n"
    )
    fichier_html_graphs.write(
        '  <object data="'
        + "sharpe_by_asset_fig.html"
        + '" width="100%" height="50%"></object>'
        + "\n"
    )
    fichier_html_graphs.write(" </div>" + "\n")

    fichier_html_graphs.write("</body></html>")
    webbrowser.open_new_tab("../result_html/DASHBOARD.html")


def get_data_and_create_graph(assets, startdate, num_portfolio, args):
    """Main function which will retrieve assets data and create graphs
    based on it
    Args:
        assets (list): Assets to analyze
        startdate (string): Date where the analyze should begin
        num_portfolio (int): number of different portfolio allocation to create
        args : optionnal command arguments
    """
    get_asset_data(assets, startdate)

    sharpe_fig, max_sharpe_port = optimize(num_portfolio)
    sharpe_by_asset_fig, ticker_to_eliminate = sharpe_each_asset_chart()
    evol_price_fig = evol_chart(max_sharpe_port.tolist()[3:], log=args.log)

    period_return_fig = period_return_chart(max_sharpe_port.tolist()[3:])

    create_final_review(
        evol_price_fig, sharpe_fig, sharpe_by_asset_fig, period_return_fig
    )
    return ticker_to_eliminate


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

    ticker_to_eliminate = get_data_and_create_graph(
        assets, startdate, num_portfolio, args
    )

    if args.filter:
        for i in range(args.filter):
            max_sharpe_port = max_sharpe_port[max_sharpe_port.values > value_filter]
            print(max_sharpe_port)
            print("############################")
            print("FILTRATION...")
            print("############################")

            new_assets = max_sharpe_port.index.tolist()[3:]
            new_assets = [x for x in new_assets if x not in ticker_to_eliminate]
            if previous_assets == new_assets: # if no filtration is made this is the "final" result
                break
            previous_assets = new_assets

            ticker_to_eliminate = get_data_and_create_graph(
                assets, startdate, num_portfolio, args
            )


if __name__ == "__main__":
    os.chdir(os.path.dirname(sys.argv[0]))
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", dest="filter", type=int, help="filtered mode")
    parser.add_argument(
        "--log", dest="log", action="store_true", help="Use logarithmic chart"
    )
    main(parser.parse_args())
