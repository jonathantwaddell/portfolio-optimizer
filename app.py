#Import Backend Functions
from backend import *
import streamlit as st

def main():
    st.set_page_config(
        page_title="Portfolio Optimizer",
        page_icon="📊",
        layout="wide"
    )

    st.title("📈 Portfolio Optimizer")
    st.write(
        "Optimize your stock portfolio using historical market data "
        "and modern portfolio theory."
    )
    tickers_raw = st.text_input("Enter Stock Symbols seperated by a space: e.g.",
                                value = "AAPL NVDA TSLA MSFT")

    time_period = st.selectbox(
        "Historical Data Period",
        [
            "1y",
            "2y",
            "5y",
            "10y",
            "ytd",
            "max"
        ]
    )

    problem_type = st.selectbox(
        "Optimization Method",
        [
            "Target Return",
            "Minimum Volatility",
            "Maximum Sharpe Ratio"
        ]
    )

    if problem_type == "Target Return":
        target = st.number_input(
            "Target Annual Return as a decimal (e.g. If you want 20%, type 0.2): ",
            min_value=0.0,
            max_value=2.0,
            value=0.10,
            step=0.01
        )

    # -------------------------
    # OPTIMIZATION
    # -------------------------

    if st.button(
            "Optimize Portfolio",
            type="primary"
    ):

        tickers = tickers_raw.upper().split()

        with st.spinner("Downloading market data..."):

            prices = get_stock_data(
                tickers,
                time_period
            )

            mu, Sigma, symbols = calculate_statistics(
                prices
            )

        with st.spinner("Optimizing portfolio..."):

            if problem_type == "Target Return":

                weights, ret, vol = optimize_portfolio(
                    mu,
                    Sigma,
                    target,
                    "return"
                )

                sharpe = portfolio_sharpe(
                    weights,
                    mu,
                    Sigma
                )


            elif problem_type == "Minimum Volatility":

                weights, ret, vol = optimize_portfolio(
                    mu,
                    Sigma,
                    None,
                    "minimum volatility"
                )

                sharpe = portfolio_sharpe(
                    weights,
                    mu,
                    Sigma
                )


            else:

                weights, ret, vol, sharpe = optimize_sharpe(
                    mu,
                    Sigma
                )

        st.session_state.optimized = True
        st.session_state.weights = weights
        st.session_state.symbols = symbols
        st.session_state.ret = ret
        st.session_state.vol = vol
        st.session_state.sharpe = sharpe
        st.session_state.mu = mu
        st.session_state.Sigma = Sigma
        st.session_state.tickers = tickers
        st.session_state.problem_type = problem_type
        st.session_state.weights_series = pd.Series(weights, index=symbols)
        st.rerun()

    if st.session_state.get("optimized"):
        weights = st.session_state.weights
        symbols = st.session_state.symbols
        ret = st.session_state.ret
        vol = st.session_state.vol
        sharpe = st.session_state.sharpe
        mu = st.session_state.mu
        Sigma = st.session_state.Sigma
        tickers = st.session_state.tickers
        problem_type = st.session_state.problem_type
        weights_series = st.session_state.weights_series

        # -------------------------
        # RESULTS
        # -------------------------

        st.divider()

        st.header("Your Optimized Portfolio")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Expected Return",
            f"{ret:.2%}"
        )

        col2.metric(
            "Expected Volatility",
            f"{vol:.2%}"
        )

        col3.metric(
            "Sharpe Ratio",
            f"{sharpe:.3f}"
        )

        # -------------------------
        # ALLOCATION
        # -------------------------

        portfolio_df = pd.DataFrame({
            "Stock": symbols,
            "Allocation": weights
        })

        portfolio_df = portfolio_df.sort_values(
            "Allocation",
            ascending=False
        )

        st.subheader("Portfolio Allocation")

        st.dataframe(
            portfolio_df.style.format({
                "Allocation": "{:.2%}"
            }),
            hide_index=True
        )

        # -------------------------
        # EFFICIENT FRONTIER
        # -------------------------

        st.subheader("Efficient Frontier")

        if problem_type == "Maximum Sharpe Ratio":

            fig = plot_efficient_frontier(
                mu,
                Sigma,
                tickers
            )

        else:

            fig = plot_efficient_frontier(
                mu,
                Sigma,
                tickers,
                (ret, vol)
            )
        corr_matrix = plot_corr_matrix(Sigma, tickers)
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(corr_matrix, use_container_width=True)

        # -------------------------
        # BACKTESTING
        # -------------------------

        st.divider()
        st.header("Backtest Portfolio")

        backtest_period = st.selectbox(
            "Backtest Period",
            ["1y", "2y", "5y", "10y", "ytd", "max"],
            key="backtest_period"
        )

        if st.button("Run Backtest", type="secondary"):

            with st.spinner("Running backtest..."):

                backtest_data = backtest_portfolio(
                    weights_series,
                    backtest_period
                )

            backtest_fig = plot_backtest(backtest_data)
            st.plotly_chart(backtest_fig, use_container_width=True)

            final_opt = backtest_data["Optimized Portfolio"].iloc[-1]
            final_equal = backtest_data["Equal Weight Portfolio"].iloc[-1]

            col1, col2 = st.columns(2)
            col1.metric(
                "Optimized Portfolio Final Value",
                f"${final_opt:.2f}"
            )
            col2.metric(
                "Equal Weight Portfolio Final Value",
                f"${final_equal:.2f}"
            )

if __name__ == '__main__':
    main()
