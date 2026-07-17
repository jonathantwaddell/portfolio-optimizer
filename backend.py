#Imports & Installs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def plot_efficient_frontier(
    mu,
    Sigma,
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

    N = 10000

    random_returns = []
    random_vols = []

    for i in range(N):

        weights = np.random.dirichlet(
            np.ones(len(mu))
        )

        random_returns.append(
            portfolio_return(weights, mu)
        )

        random_vols.append(
            portfolio_vol(weights, Sigma)
        )

    fig = plt.figure(figsize=(8, 6))

    plt.scatter(
        random_vols,
        random_returns,
        s=3,
        alpha=0.3,
        label="Random Portfolios"
    )

    plt.plot(
        front_vols,
        front_returns,
        color="red",
        linewidth=3,
        label="Efficient Frontier"
    )

    # Tangency portfolio
    plt.scatter(
        t_vol,
        t_return,
        s=100,
        marker='*',
        label="Maximum Sharpe Portfolio"
    )

    # Capital Market Line
    plt.plot(
        vols,
        cml_returns,
        '--',
        linewidth=2,
        label="Capital Market Line"
    )

    if your_port is not None:

        plt.scatter(
            your_port[1],
            your_port[0],
            s=100,
            marker='x',
            label="Your Portfolio"
        )

    plt.legend()

    plt.xlabel("Volatility")
    plt.ylabel("Expected Return")

    plt.title("Efficient Frontier")

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

