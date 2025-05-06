import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- Broker converters ---

def convert_zerodha_to_standard(file_buffer, account=""):
    df = pd.read_excel(file_buffer, skiprows=14)
    df.columns = df.columns.str.strip()
    df['Trade Date'] = pd.to_datetime(df['Trade Date'])
    output_df = pd.DataFrame()
    output_df['TradeID'] = df['Trade ID'].astype(str)
    output_df['TradeDate'] = df['Trade Date'].dt.date
    output_df['SettlementDate'] = output_df['TradeDate'].apply(lambda x: x + timedelta(days=2))
    output_df['Broker'] = 'Zerodha'
    output_df['Account'] = account
    output_df['Symbol'] = df['Symbol']
    output_df['ISIN'] = df['ISIN']
    output_df['BuySell'] = df['Trade Type'].str.lower()
    output_df['Quantity'] = df['Quantity']
    output_df['Price'] = df['Price']
    output_df['Currency'] = 'INR'
    output_df['NotionalValue'] = output_df['Quantity'] * output_df['Price']
    output_df['Commission'] = ''
    output_df['TradeCurrency'] = 'INR'
    output_df['Exchange'] = df['Exchange']
    output_df['TradeExecutionTime'] = df['Order Execution Time']
    return output_df

broker_conversion_map = {
    "Zerodha": convert_zerodha_to_standard,
    # Add other brokers here
}

# --------------- Portfolio construction -------------------

def build_portfolio_timeseries(trades_df):
    trades_df.columns = trades_df.columns.str.strip()
    trades_df['TradeDate'] = pd.to_datetime(trades_df['TradeDate'])
    trades_df['BuySell'] = trades_df['BuySell'].str.lower()
    trades_df['SignedQty'] = trades_df.apply(lambda x: x['Quantity'] if x['BuySell'] == 'buy' else -x['Quantity'], axis=1)

    start_date = trades_df['TradeDate'].min()
    end_date = pd.to_datetime(datetime.today().date())

    tickers = trades_df['Symbol'].unique()
    all_dates = pd.date_range(start_date, end_date, freq='B')
    holdings_daily = pd.DataFrame(index=all_dates)

    for ticker in tickers:
        ticker_trades = trades_df[trades_df['Symbol'] == ticker]
        daily_trades = ticker_trades.groupby('TradeDate')['SignedQty'].sum()
        daily_trades = daily_trades.reindex(all_dates, fill_value=0)
        holdings_daily[ticker] = daily_trades.cumsum()

    yf_tickers = [t if t.endswith('.NS') else t + '.NS' for t in tickers]
    data = yf.download(yf_tickers, start=start_date, end=end_date)

    prices_df = data['Adj Close'] if 'Adj Close' in data else data['Close']
    if isinstance(prices_df, pd.Series):
        prices_df = prices_df.to_frame()

    prices_df = prices_df.reindex(all_dates).ffill().bfill()
    prices_df.columns = [c.replace('.NS', '') for c in prices_df.columns]

    portfolio_value = (holdings_daily * prices_df).sum(axis=1)
    portfolio_returns = portfolio_value.pct_change().dropna()
    portfolio_returns.index = portfolio_returns.index.tz_localize('UTC')

    return portfolio_returns, portfolio_value, holdings_daily, prices_df

# --------------- Benchmark Returns ---------------------

def fetch_benchmark_returns(ticker, start_date, end_date):
    prices = yf.download(ticker, start=start_date, end=end_date)['Close'].dropna()
    returns = prices.pct_change().dropna()
    returns.index = returns.index.tz_localize('UTC')
    return returns

# --------------- Format Per-stock Performance ----------------

def format_per_stock_metrics(df):
    df_formatted = df.copy()
    fmt_pct = lambda x: f"{x:.2%}" if pd.notnull(x) and np.isfinite(x) else "N/A"
    fmt_float = lambda x: f"{x:.2f}" if pd.notnull(x) and np.isfinite(x) else "N/A"
    fmt_int = lambda x: f"{int(x)}" if pd.notnull(x) else "0"

    if 'Max Drawdown' in df_formatted.columns:
        df_formatted['Max Drawdown'] = df_formatted['Max Drawdown'].map(fmt_pct)
    df_formatted['Annual Volatility'] = df_formatted['Annual Volatility'].map(fmt_pct)
    df_formatted['Annualized Return'] = df_formatted['Annualized Return'].map(fmt_pct)
    df_formatted['Sharpe Ratio'] = df_formatted['Sharpe Ratio'].map(fmt_float)
    df_formatted['Realized P&L'] = df_formatted['Realized P&L'].map(fmt_float)
    df_formatted['Unrealized P&L'] = df_formatted['Unrealized P&L'].map(fmt_float)
    df_formatted['Total P&L'] = df_formatted['Total P&L'].map(fmt_float)
    df_formatted['Quantity'] = df_formatted['Quantity'].map(fmt_int)
    df_formatted['Buy Price'] = df_formatted['Buy Price'].map(fmt_float)
    df_formatted['Current Price'] = df_formatted['Current Price'].map(fmt_float)

    return df_formatted

# --------------- FIFO P&L Calculation ---------------------

def calculate_fifo_pnl(trades_df, holdings_daily, prices_df):
    data = []
    for symbol in holdings_daily.columns:
        fifo_queue = []
        realized_pnl = 0.0
        trades_sym = trades_df[trades_df['Symbol'] == symbol].sort_values('TradeDate')

        for _, trade in trades_sym.iterrows():
            qty = trade['Quantity']
            price = trade['Price']
            side = trade['BuySell'].lower()

            if side == 'buy':
                fifo_queue.append({'qty': qty, 'price': price})
            elif side == 'sell':
                sell_qty = qty
                sell_price = price

                while sell_qty > 0 and fifo_queue:
                    lot = fifo_queue[0]
                    if lot['qty'] <= sell_qty:
                        realized_pnl += (sell_price - lot['price']) * lot['qty']
                        sell_qty -= lot['qty']
                        fifo_queue.pop(0)
                    else:
                        realized_pnl += (sell_price - lot['price']) * sell_qty
                        lot['qty'] -= sell_qty
                        sell_qty = 0

                if sell_qty > 0:
                    # Unusual case: selling more than holding, treat as zero cost basis
                    realized_pnl += sell_price * sell_qty

        unrealized_cost = sum(lot['qty'] * lot['price'] for lot in fifo_queue)
        unrealized_qty = sum(lot['qty'] for lot in fifo_queue)

        current_price = prices_df[symbol].iloc[-1] if symbol in prices_df.columns else np.nan

        unrealized_value = unrealized_qty * current_price if not pd.isna(current_price) else np.nan
        unrealized_pnl = unrealized_value - unrealized_cost if not pd.isna(unrealized_value) else np.nan

        total_pnl = realized_pnl + (unrealized_pnl if not np.isnan(unrealized_pnl) else 0.0)
        avg_buy_price = unrealized_cost / unrealized_qty if unrealized_qty > 0 else np.nan
        quantity = holdings_daily[symbol].iloc[-1] if symbol in holdings_daily.columns else 0.0

        vals = holdings_daily[symbol] * prices_df.get(symbol, pd.Series(dtype=float))
        returns = vals.pct_change().dropna()
        ann_vol = returns.std() * np.sqrt(252) if len(returns) > 1 else np.nan

        length = len(returns)
        total_return = vals.iloc[-1] / vals.iloc[0] if len(vals) > 1 else np.nan

        if vals.iloc[0] == 0 or length < 2 or np.isnan(total_return):
            ann_return = np.nan
        else:
            ann_return = (total_return ** (252 / length)) - 1

        sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

        if len(returns) > 0:
            cumulative = (1 + returns).cumprod()
            peak = cumulative.cummax()
            drawdown = (cumulative - peak) / peak
            max_drawdown = drawdown.min()
        else:
            max_drawdown = np.nan

        data.append({
            'Stock': symbol,
            'Quantity': quantity,
            'Buy Price': avg_buy_price,
            'Current Price': current_price,
            'Realized P&L': realized_pnl,
            'Unrealized P&L': unrealized_pnl,
            'Total P&L': total_pnl,
            'Max Drawdown': max_drawdown,
            'Annual Volatility': ann_vol,
            'Annualized Return': ann_return,
            'Sharpe Ratio': sharpe
        })

    return pd.DataFrame(data)

# -------------- Plot Helpers ------------------

def plot_line(series, title, ylabel):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(series)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

def plot_bar(series, title, ylabel):
    fig, ax = plt.subplots(figsize=(10,5))
    series.plot.bar(ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

def plot_hist(series, title, bins=50):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.hist(series, bins=bins, color='blue', alpha=0.7, edgecolor='black')
    ax.set_title(title)
    ax.set_ylabel('Frequency')
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

# -------------- Sector Allocation -------------------

def sector_allocation(holdings_daily, prices_df):
    import yfinance as yf
    st.subheader("Sector Allocation")

    latest_holdings = holdings_daily.iloc[-1]
    latest_prices = prices_df.iloc[-1]
    market_values = latest_holdings * latest_prices

    sectors = {}
    for symbol in latest_holdings.index:
        try:
            info = yf.Ticker(symbol + ".NS").info
            sectors[symbol] = info.get("sector", "Unknown")
        except:
            sectors[symbol] = "Unknown"

    df = pd.DataFrame({
        "Symbol": latest_holdings.index,
        "Sector": [sectors[s] for s in latest_holdings.index],
        "MarketValue": market_values,
    })

    sector_dist = df.groupby("Sector")["MarketValue"].sum()
    sector_alloc = sector_dist / sector_dist.sum()

    if sector_alloc.empty:
        st.warning("No sector allocation data available.")
    else:
        st.pyplot(sector_alloc.plot.pie(autopct="%1.1f%%", figsize=(6, 6)).figure)
    st.write("### Sector Allocation Table")
    st.dataframe(sector_dist.sort_values(ascending=False).to_frame("Market Value"))

# -------------- Trade History ---------------------

def show_trade_history(trades_df):
    st.subheader("Trade History")
    st.dataframe(trades_df.sort_values(['TradeDate', 'Symbol']))

# -------------- P&L Curve ---------------------

def show_pnl_curve(portfolio_value):
    st.subheader("Portfolio Daily P&L")
    pnl_curve = portfolio_value.diff().fillna(0)
    plot_line(pnl_curve, "Daily P&L", "P&L")

# -------------- Benchmark and advanced analytics -----------------------

def compute_drawdowns(returns):
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    max_dd = drawdown.min()
    return drawdown, max_dd

def rolling_volatility(returns, window=30):
    return returns.rolling(window=window).std() * np.sqrt(252)

def rolling_sharpe_ratio(returns, window=30, riskfree_rate=0):
    mean = returns.rolling(window).mean() * 252
    std = returns.rolling(window).std() * np.sqrt(252)
    sharpe = (mean - riskfree_rate) / std
    return sharpe
def compute_excess_returns(portfolio_rets, benchmark_rets):
    idx = portfolio_rets.index.intersection(benchmark_rets.index)
    port = portfolio_rets.loc[idx]
    bench = benchmark_rets.loc[idx]
    return port - bench, port, bench

def rolling_beta(portfolio_rets, benchmark_rets, window=126):
    cov = portfolio_rets.rolling(window).cov(benchmark_rets)
    var = benchmark_rets.rolling(window).var()
    return cov / var

def rolling_tracking_error(portfolio_rets, benchmark_rets, window=126):
    diff = portfolio_rets - benchmark_rets
    return diff.rolling(window).std() * np.sqrt(252)

def rolling_alpha(portfolio_rets, benchmark_rets, riskfree_rate=0, window=126):
    beta_series = rolling_beta(portfolio_rets, benchmark_rets, window)
    excess_rets = portfolio_rets - benchmark_rets
    ann_excess_return = excess_rets.rolling(window).mean() * 252
    return ann_excess_return - beta_series * 0

# ------------ VaR and Marginal VaR -----------------

def historical_var(returns, confidence_level=0.95):
    var = np.percentile(returns.dropna(), 100 * (1 - confidence_level))
    return var

def plot_var_graph(portfolio_returns, confidence_level=0.95):
    losses = -portfolio_returns.dropna()
    var_val = np.percentile(losses, 100 * confidence_level)
    plt.figure(figsize=(10,6))
    plt.hist(losses, bins=50, alpha=0.75, color='skyblue', edgecolor='black')
    plt.axvline(var_val, color='red', linestyle='dashed', linewidth=2, label=f'{int(confidence_level*100)}% VaR = {var_val:.4f}')
    plt.title(f'Portfolio Loss Distribution & {int(confidence_level*100)}% VaR')
    plt.xlabel('Portfolio Loss')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt.gcf())
    plt.close()

def marginal_var(holdings_daily, prices_df, portfolio_returns, confidence_level=0.95):
    portfolio_value = (holdings_daily * prices_df).sum(axis=1)
    portfolio_losses = -portfolio_value.pct_change().dropna()
    var_threshold = np.percentile(portfolio_losses, confidence_level * 100)
    worst_days = portfolio_losses[portfolio_losses >= var_threshold].index
    marginals = {}
    for stock in holdings_daily.columns:
        stock_values = holdings_daily[stock] * prices_df[stock]
        stock_losses = -stock_values.pct_change().dropna()
        common_days = stock_losses.index.intersection(worst_days)
        if len(common_days) == 0:
            marginals[stock] = 0.0
        else:
            marginals[stock] = stock_losses.loc[common_days].mean()
    marginals_series = pd.Series(marginals)
    total = marginals_series.abs().sum()
    if total > 0:
        marginals_series = marginals_series / total
    else:
        marginals_series[:] = 0.0
    return marginals_series


# ------------ Main Function ------------------------

def main():
    st.title("Comprehensive Equity Portfolio Dashboard")

    broker = st.selectbox("Select Broker", list(broker_conversion_map.keys()))

    uploaded_file = st.file_uploader("Upload your broker trade Excel file", type=["xlsx"])
    account_id = st.text_input("Enter Account ID", value="ACC01")

    if uploaded_file and broker:
        try:
            converter = broker_conversion_map[broker]
            trades_df = converter(uploaded_file, account=account_id)
            st.success(f"Trades converted from {broker} format.")
            st.dataframe(trades_df.head())

            portfolio_returns, portfolio_value, holdings_daily, prices_df = build_portfolio_timeseries(trades_df)
            benchmark_returns = fetch_benchmark_returns("^NSEI", portfolio_returns.index.min(), portfolio_returns.index.max())

            if isinstance(portfolio_returns, pd.DataFrame):
                portfolio_returns = portfolio_returns.iloc[:, 0]
            if isinstance(benchmark_returns, pd.DataFrame):
                benchmark_returns = benchmark_returns.iloc[:, 0]

            idx = portfolio_returns.index.intersection(benchmark_returns.index)
            portfolio_returns = portfolio_returns.loc[idx]
            benchmark_returns = benchmark_returns.loc[idx]

            section = st.sidebar.selectbox("Select Analysis Section", [
                "Basic Analytics",
                "Advanced Analytics",
                "Other Analytics",
                "Round Trip Tear Sheet"
            ])

            st.write("### Portfolio Value Over Time")
            st.line_chart(portfolio_value)

            if section == "Basic Analytics":
                st.subheader("Basic Portfolio Analytics")
                plot_line((1 + portfolio_returns).cumprod() -1, "Cumulative Returns", "Cumulative Return")
                drawdown, max_dd = compute_drawdowns(portfolio_returns)
                st.write(f"Max Drawdown: {max_dd:.2%}")
                plot_line(drawdown, "Drawdowns", "Drawdown")
                plot_line(rolling_volatility(portfolio_returns), "Rolling Volatility (30-day)", "Volatility")
                plot_line(rolling_sharpe_ratio(portfolio_returns), "Rolling Sharpe Ratio (30-day)", "Sharpe Ratio")
                plot_bar(portfolio_returns.resample('YE').apply(lambda x:(1+x).prod()-1), "Annual Returns", "Return")
                plot_hist(portfolio_returns, "Distribution of Daily Returns")
                perf_stats = {
                    "Annualized Return": ((1 + portfolio_returns).prod() - 1) ** (252 / len(portfolio_returns)) - 1,
                    "Annualized Volatility": portfolio_returns.std() * np.sqrt(252),
                    "Sharpe Ratio": (((1 + portfolio_returns).prod() - 1) ** (252 / len(portfolio_returns)) - 1) /
                                    (portfolio_returns.std() * np.sqrt(252)),
                }
                st.subheader("Performance Summary")
                st.table({k: f"{v:.2%}" for k,v in perf_stats.items()})

                per_stock_df = calculate_fifo_pnl(trades_df, holdings_daily, prices_df)
                st.subheader("Per-Stock Performance Metrics")
                st.dataframe(format_per_stock_metrics(per_stock_df).sort_values("Total P&L", ascending=False))

            elif section == "Advanced Analytics":
                st.subheader("Advanced Portfolio Analytics")
                excess, port_rets, bench_rets = compute_excess_returns(portfolio_returns, benchmark_returns)
                beta_series = rolling_beta(port_rets, bench_rets)
                tracking_err = rolling_tracking_error(port_rets, bench_rets)
                alpha_series = rolling_alpha(port_rets, bench_rets)
                plot_line(beta_series, "Rolling Beta (126-day)", "Beta")
                plot_line(tracking_err, "Rolling Tracking Error (126-day)", "Tracking Error")
                plot_line(alpha_series, "Rolling Alpha (126-day)", "Alpha")
                st.subheader("Value at Risk (VaR)")
                var_95 = historical_var(portfolio_returns)
                st.write(f"95% Historical VaR: {var_95:.4%}")
                st.write("Portfolio Loss Distribution & VaR")
                plot_var_graph(portfolio_returns)

                marginal_series = marginal_var(holdings_daily, prices_df, portfolio_returns)
                st.subheader("Marginal VaR Contributions")
                st.bar_chart(marginal_series.sort_values(ascending=False))

            elif section == "Other Analytics":
                st.subheader("Sector Allocation")
                sector_allocation(holdings_daily, prices_df)
                show_trade_history(trades_df)
                show_pnl_curve(portfolio_value)
                        
        except Exception as ex:
            st.error(f"An error occurred: {ex}")

if __name__ == "__main__":
    main()
