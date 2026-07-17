#Imports & Installs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
from scipy.optimize import minimize
import yfinance as yf

#Portfolio Funcitons
def portfolio_return(w, mu):
    return w @ mu

def portfolio_variance(w, cov):
    return w @ cov @ w

def portfolio_vol(w, cov):
    return np.sqrt(portfolio_variance(w, cov))

def portfolio_sharpe(w, mu, Sigma, risk_free=0.0455):
    ret = portfolio_return(w, mu)
    vol = portfolio_vol(w, Sigma)
    return (ret - risk_free) / vol

#Objective Functions
def objective_v(w, Sigma):
    return portfolio_variance(w,Sigma)

def objective_d(w, mu):
    return -portfolio_return(w,mu)

def objective_sharpe(w, mu, Sigma, risk_free=0.0455):
    return -portfolio_sharpe(w, mu, Sigma, risk_free)

##Constraint equations
def weight_constraint(w):
    return np.sum(w)-1

def return_constraint(w, mu, target_return):
    return portfolio_return(w, mu) - target_return

def variance_constraint(w, cov, max_variance):
    return max_variance - portfolio_variance(w, cov)

#---------------------------
#Optimization Functions
#---------------------------
def optimize_sharpe(mu, Sigma, risk_free=0.0455):
  ##Optimizing potfolio to have the highest possible Sharpe Ratio

    n = len(mu)

    w0 = np.ones(n) / n

    bounds = [(0,1)] * n

    constraints = [
        {
            'type': 'eq',
            'fun': weight_constraint
        }
    ]

    result = minimize(
        objective_sharpe,
        w0,
        args=(mu, Sigma, risk_free),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    weights = result.x

    ret = portfolio_return(weights, mu)
    vol = portfolio_vol(weights, Sigma)
    sharpe = portfolio_sharpe(weights, mu, Sigma, risk_free)

    return weights, ret, vol, sharpe


def optimize_portfolio(mu, Sigma, target, problem_type):
    ##Optimizing given a target 'return' or a miniimum target 'minimum volatitlity'

    n = len(mu)

    w0 = np.ones(n) / n

    bounds = [(0, 1)] * n

    unconstrained = minimize(
        objective_v,
        w0,
        args=(Sigma,),
        method="SLSQP",
        bounds=bounds,
        constraints=[{'type': 'eq', 'fun': weight_constraint}]
    )

    minimum_vol = portfolio_vol(unconstrained.x, Sigma)

    if problem_type == 'minimum volatility':
        weights = unconstrained.x
        return weights, portfolio_return(weights, mu), portfolio_vol(weights, Sigma)

    if problem_type == "return":
        # User specifies a target return
        target_return = target
        max_variance = None

    else:
        raise ValueError("Problem type must be 'return' or 'minimum volatility'.")

        max_variance = target ** 2

    constraints_drift = [
        {
            'type': 'eq',
            'fun': weight_constraint
        },
        {
            'type': 'ineq',
            'fun': return_constraint,
            'args': (mu, target_return)
        }
    ]

    if problem_type == "return":
        # User specifies a target return
        objective = objective_v  # minimize variance
        objective_args = (Sigma,)
        constraints = constraints_drift

    result = minimize(objective, w0, args=objective_args, method='SLSQP', bounds=bounds, constraints=constraints)
    # print(result.success)
    # print(result.message)
    weights = result.x

    return weights, portfolio_return(weights, mu), portfolio_vol(weights, Sigma)

def efficient_frontier(mu, Sigma, n_points=50):

    _, min_ret, min_vol = optimize_portfolio(
    mu,
    Sigma,
    0,
    "return"
    )

    max_return = max(mu)

    target_returns = np.linspace(min_ret, max_return, n_points)

    frontier_returns = []
    frontier_vols = []
    frontier_weights = []

    for target in target_returns:

        try:
            weights, ret, vol = optimize_portfolio(
                mu,
                Sigma,
                target,
                "return"
            )

            frontier_returns.append(ret)
            frontier_vols.append(vol)
            frontier_weights.append(weights)

        except:
            pass

    return frontier_returns, frontier_vols, frontier_weights

import numpy as np
import plotly.graph_objects as go


import numpy as np
import plotly.graph_objects as go


def plot_efficient_frontier(
    mu,
    Sigma,
    symbols,
    your_port=None,
    risk_free_rate=0.0455
):

    front_returns, front_vols, front_weights = efficient_frontier(
        mu,
        Sigma
    )

    t_weights, t_return, t_vol, t_sharpe = optimize_sharpe(
        mu,
        Sigma,
        risk_free_rate
    )

    vols = np.linspace(
        0,
        max(front_vols) * 1.1,
        100
    )

    cml_returns = risk_free_rate + t_sharpe * vols

    # -----------------------------
    # Generate random portfolios
    # -----------------------------

    N = 10000

    random_returns = []
    random_vols = []
    random_weights = []

    for _ in range(N):

        weights = np.random.dirichlet(
            np.ones(len(mu))
        )

        random_returns.append(
            portfolio_return(weights, mu)
        )

        random_vols.append(
            portfolio_vol(weights, Sigma)
        )

        random_weights.append(weights)

    # -----------------------------
    # Hover text for random portfolios
    # -----------------------------

    random_hover_text = []

    for weights in random_weights:

        allocation_text = "<br>".join(
            f"{symbol}: {weight:.2%}"
            for symbol, weight in zip(symbols, weights)
        )

        random_hover_text.append(
            allocation_text
        )

    # -----------------------------
    # Hover text for efficient frontier
    # -----------------------------

    frontier_hover_text = []

    for weights in front_weights:

        allocation_text = "<br>".join(
            f"{symbol}: {weight:.2%}"
            for symbol, weight in zip(symbols, weights)
        )

        frontier_hover_text.append(
            allocation_text
        )

    # -----------------------------
    # Plotly figure
    # -----------------------------

    fig = go.Figure()

    # Random portfolios
    fig.add_trace(
        go.Scatter(
            x=random_vols,
            y=random_returns,
            mode="markers",
            name="Random Portfolios",

            marker=dict(
                size=4,
                opacity=0.35
            ),

            text=random_hover_text,

            hovertemplate=(
                "<b>Random Portfolio</b><br>"
                "Volatility: %{x:.2%}<br>"
                "Expected Return: %{y:.2%}<br>"
                "<br>"
                "<b>Portfolio Allocation:</b><br>"
                "%{text}"
                "<extra></extra>"
            )
        )
    )

    # Efficient frontier
    fig.add_trace(
        go.Scatter(
            x=front_vols,
            y=front_returns,
            mode="lines+markers",
            name="Efficient Frontier",

            marker=dict(
                size=5
            ),

            line=dict(
                width=3
            ),

            text=frontier_hover_text,

            hovertemplate=(
                "<b>Efficient Frontier</b><br>"
                "Volatility: %{x:.2%}<br>"
                "Expected Return: %{y:.2%}<br>"
                "<br>"
                "<b>Portfolio Allocation:</b><br>"
                "%{text}"
                "<extra></extra>"
            )
        )
    )

    # Maximum Sharpe portfolio
    t_allocation_text = "<br>".join(
        f"{symbol}: {weight:.2%}"
        for symbol, weight in zip(symbols, t_weights)
    )

    fig.add_trace(
        go.Scatter(
            x=[t_vol],
            y=[t_return],
            mode="markers",
            name="Maximum Sharpe Portfolio",

            marker=dict(
                size=16,
                symbol="star"
            ),

            hovertemplate=(
                "<b>Maximum Sharpe Portfolio</b><br>"
                "Volatility: %{x:.2%}<br>"
                "Expected Return: %{y:.2%}<br>"
                f"Sharpe Ratio: {t_sharpe:.3f}<br>"
                "<br>"
                "<b>Portfolio Allocation:</b><br>"
                f"{t_allocation_text}"
                "<extra></extra>"
            )
        )
    )

    # Capital Market Line
    fig.add_trace(
        go.Scatter(
            x=vols,
            y=cml_returns,
            mode="lines",
            name="Capital Market Line",

            line=dict(
                dash="dash",
                width=2
            ),

            hovertemplate=(
                "Volatility: %{x:.2%}<br>"
                "Expected Return: %{y:.2%}"
                "<extra></extra>"
            )
        )
    )

    # User portfolio
    if your_port is not None:

        your_return, your_vol = your_port

        fig.add_trace(
            go.Scatter(
                x=[your_vol],
                y=[your_return],
                mode="markers",
                name="Your Portfolio",

                marker=dict(
                    size=16,
                    symbol="x"
                ),

                hovertemplate=(
                    "<b>Your Portfolio</b><br>"
                    "Volatility: %{x:.2%}<br>"
                    "Expected Return: %{y:.2%}"
                    "<extra></extra>"
                )
            )
        )

    # -----------------------------
    # Figure formatting
    # -----------------------------

    fig.update_layout(

        title="Efficient Frontier",

        xaxis_title="Volatility",

        yaxis_title="Expected Return",

        xaxis=dict(
            tickformat=".0%"
        ),

        yaxis=dict(
            tickformat=".0%"
        ),

        hovermode="closest",

        template="plotly_white",

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),

        margin=dict(
            l=60,
            r=30,
            t=80,
            b=60
        )
    )

    return fig

def plot_corr_matrix(Sigma, symbols):

    std = np.sqrt(np.diag(Sigma))

    corr = Sigma / np.outer(std, std)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr,
            x=symbols,
            y=symbols,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            zmid=0,

            text=np.round(corr, 2),
            texttemplate="%{text}",

            hovertemplate=(
                "<b>%{x} vs %{y}</b><br>"
                "Correlation: %{z:.3f}"
                "<extra></extra>"
            ),

            colorbar=dict(
                title="Correlation"
            )
        )
    )

    fig.update_layout(
        title=dict(
            text="Asset Correlation Matrix",
            x=0.5,
            xanchor="center"
        ),

        xaxis=dict(
            title="Asset",
            side="top"
        ),

        yaxis=dict(
            title="Asset",
            autorange="reversed"
        ),

        template="plotly_white",

        margin=dict(
            l=80,
            r=40,
            t=100,
            b=80
        )
    )

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1
    )

    return fig
#---------------------------------
#Data Collection Functions
#--------------------------------

def get_stock_data(
    tickers,
    period,
    interval="1d"
):

    prices = yf.download(
        tickers,
        period=period,
        interval=interval,
        auto_adjust=True
    )["Close"]

    return prices


def calculate_statistics(prices):

    returns = prices.pct_change().dropna()

    mu = returns.mean() * 252

    Sigma = returns.cov() * 252

    return mu.to_numpy(), Sigma.to_numpy(), list(prices.columns)

