import os
import time
import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from binance.spot import Spot as Client
from bitso import Api as BitsoClient
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional
from colorama import init, Fore, Style
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize colorama for colored console output
init()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

# Constants
ORDER_TYPE_MARKET = os.getenv("ORDER_TYPE_MARKET")
SIDE_BUY = os.getenv("SIDE_BUY")
SIDE_SELL = os.getenv("SIDE_SELL")
INTERVAL = os.getenv("INTERVAL")  # Changed to 15 minute interval
LIMIT = int(os.getenv("LIMIT"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD"))
SMA_FAST = int(os.getenv("SMA_FAST"))
SMA_SLOW = int(os.getenv("SMA_SLOW"))
RSI_OVERBOUGHT = int(os.getenv("RSI_OVERBOUGHT"))
RSI_OVERSOLD = int(os.getenv("RSI_OVERSOLD"))
FEE_RATE = float(os.getenv("FEE_RATE"))  # 0.5% fee buffer
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL"))  # 15 minutes in seconds
STOP_LOSS = float(os.getenv("STOP_LOSS"))  # 2% stop loss
TAKE_PROFIT = float(os.getenv("TAKE_PROFIT"))  # 5% take profit
MONTHLY_TARGET = float(os.getenv("MONTHLY_TARGET"))  # 5% monthly target

# Validate constants
if ORDER_TYPE_MARKET not in ["MARKET", "LIMIT"]:
    raise ValueError("ORDER_TYPE_MARKET must be 'MARKET' or 'LIMIT'")

if SIDE_BUY not in ["BUY", "SELL"]:
    raise ValueError("SIDE_BUY must be 'BUY' or 'SELL'")

if SIDE_SELL not in ["BUY", "SELL"]:
    raise ValueError("SIDE_SELL must be 'BUY' or 'SELL'")

if INTERVAL not in [
    "1m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "3d",
    "1w",
    "1M",
]:
    raise ValueError("INTERVAL must be a valid time frame")

if LIMIT < 1:
    raise ValueError("LIMIT must be a positive integer")

if RSI_PERIOD < 1:
    raise ValueError("RSI_PERIOD must be a positive integer")

if SMA_FAST < 1:
    raise ValueError("SMA_FAST must be a positive integer")

if SMA_SLOW < 1:
    raise ValueError("SMA_SLOW must be a positive integer")

if RSI_OVERBOUGHT < 0 or RSI_OVERBOUGHT > 100:
    raise ValueError("RSI_OVERBOUGHT must be between 0 and 100")

if RSI_OVERSOLD < 0 or RSI_OVERSOLD > 100:
    raise ValueError("RSI_OVERSOLD must be between 0 and 100")

if FEE_RATE < 0 or FEE_RATE > 1:
    raise ValueError("FEE_RATE must be between 0 and 1")

if CHECK_INTERVAL < 1:
    raise ValueError("CHECK_INTERVAL must be a positive integer")

if STOP_LOSS > 0 or STOP_LOSS < -1:
    raise ValueError("STOP_LOSS must be between -1 and 0")

if TAKE_PROFIT < 0 or TAKE_PROFIT > 1:
    raise ValueError("TAKE_PROFIT must be between 0 and 1")

if MONTHLY_TARGET < 0 or MONTHLY_TARGET > 1:
    raise ValueError("MONTHLY_TARGET must be between 0 and 1")

# Exchange Configuration
EXCHANGE = os.getenv("EXCHANGE")  # BINANCE or BITSO
MAX_BUY_USD = float(os.getenv("MAX_BUY_USD"))  # Max USD to spend per trade
MAX_BUY_MXN = float(os.getenv("MAX_BUY_MXN"))  # Max MXN to spend per trade
MAX_SELL_USD = float(os.getenv("MAX_SELL_USD"))  # Max USD to sell per trade
MAX_SELL_MXN = float(os.getenv("MAX_SELL_MXN"))  # Max MXN to sell per trade

# Validate exchange configuration
if EXCHANGE not in ["BINANCE", "BITSO"]:
    raise ValueError("EXCHANGE must be 'BINANCE' or 'BITSO'")

if MAX_BUY_USD < 0:
    raise ValueError("MAX_BUY_USD must be a non-negative number")

if MAX_BUY_MXN < 0:
    raise ValueError("MAX_BUY_MXN must be a non-negative number")

if MAX_SELL_USD < 0:
    raise ValueError("MAX_SELL_USD must be a non-negative number")

if MAX_SELL_MXN < 0:
    raise ValueError("MAX_SELL_MXN must be a non-negative number")


class State:
    def __init__(self):
        self.saldo_money: float = float(os.getenv("INITIAL_BALANCE"))
        self.monedas: float = 0.0
        self.is_simulation: bool = os.getenv("IS_SIMULATION", "True") == "True"
        self.last_price: float = 0.0
        self.trades_history: List[Dict] = []
        self.total_profit: float = 0.0
        self.total_losses: float = 0.0
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.losing_trades: int = 0
        self.monthly_profit: float = 0.0
        self.month_start_balance: float = self.saldo_money
        self.last_month: int = datetime.now().month
        self.currency = "USD" if EXCHANGE == "BINANCE" else "MXN"


class TradingBot:
    def __init__(self):
        if EXCHANGE == "BINANCE":
            self.api_key = os.getenv("BINANCE_API_KEY")
            self.api_secret = os.getenv("BINANCE_API_SECRET")
            self.client = Client(self.api_key, self.api_secret)
            self.symbol = "BTCUSDT"
        else:  # BITSO
            self.api_key = os.getenv("BITSO_API_KEY", "")
            self.api_secret = os.getenv("BITSO_API_SECRET", "")
            self.client = BitsoClient(self.api_key, self.api_secret)
            self.symbol = "btc_mxn"

        self.state = State()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.max_buy = MAX_BUY_USD if EXCHANGE == "BINANCE" else MAX_BUY_MXN
        self.max_sell = MAX_SELL_USD if EXCHANGE == "BINANCE" else MAX_SELL_MXN

        init_msg = (
            "\n"
            + "=" * 50
            + "\n"
            + f"Bot initialized - Trading {self.symbol} on {EXCHANGE} in {'simulation' if self.state.is_simulation else 'live'} mode\n"
            + f"Max buy per trade: {self.max_buy} {self.state.currency}\n"
            + f"Max sell per trade: {self.max_sell} {self.state.currency}\n"
            + "=" * 50
        )
        print(f"{Fore.CYAN}{init_msg}{Style.RESET_ALL}")
        logger.info(init_msg)

    def get_historical_data(self) -> pd.DataFrame:
        try:
            if EXCHANGE == "BINANCE":
                klines = self.client.klines(
                    symbol=self.symbol, interval=INTERVAL, limit=LIMIT
                )

                columns = [
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_asset_volume",
                    "number_of_trades",
                    "taker_buy_base_asset_volume",
                    "taker_buy_quote_asset_volume",
                    "ignore",
                ]

                df = pd.DataFrame(klines, columns=columns)
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            else:  # BITSO
                trades = self.client.get_ohlcv(self.symbol, INTERVAL, LIMIT)
                df = pd.DataFrame(trades)
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

            df.set_index("timestamp", inplace=True)
            df = df.astype(float)
            return df

        except Exception as e:
            error_msg = (
                "\n" + "!" * 50 + f"\nHistorical data error: {str(e)}\n" + "!" * 50
            )
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
            logger.error(error_msg)
            return pd.DataFrame()

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            futures = []
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures.append(
                    executor.submit(lambda: df["close"].rolling(window=SMA_FAST).mean())
                )
                futures.append(
                    executor.submit(lambda: df["close"].rolling(window=SMA_SLOW).mean())
                )
                futures.append(executor.submit(lambda: self.calculate_rsi(df["close"])))

            df["SMA20"], df["SMA50"], df["RSI"] = [f.result() for f in futures]

            df["Buy_Signal"] = pd.Series([None] * len(df))
            df["Sell_Signal"] = pd.Series([None] * len(df))

            buy_conditions = (
                (df["SMA20"] > df["SMA50"])
                & (df["SMA20"].shift(1) <= df["SMA50"].shift(1))
                & (df["RSI"] < RSI_OVERBOUGHT)
            )

            sell_conditions = (
                (df["SMA20"] < df["SMA50"])
                & (df["SMA20"].shift(1) >= df["SMA50"].shift(1))
                & (df["RSI"] > RSI_OVERSOLD)
            )

            df.loc[buy_conditions, "Buy_Signal"] = df["close"]
            df.loc[sell_conditions, "Sell_Signal"] = df["close"]

            # Log signal analysis
            if buy_conditions.iloc[-1]:
                signal_msg = (
                    "\n" + "-" * 50 + "\n"
                    "SIGNAL ANALYSIS:\n"
                    "SMA20 crossed above SMA50 (Bullish)\n"
                    f"RSI: {df['RSI'].iloc[-1]:.2f} (Not overbought)\n"
                    "Potential BUY signal detected\n" + "-" * 50
                )
                print(f"{Fore.CYAN}{signal_msg}{Style.RESET_ALL}")
                logger.info(signal_msg)
            elif sell_conditions.iloc[-1]:
                signal_msg = (
                    "\n" + "-" * 50 + "\n"
                    "SIGNAL ANALYSIS:\n"
                    "SMA20 crossed below SMA50 (Bearish)\n"
                    f"RSI: {df['RSI'].iloc[-1]:.2f} (Not oversold)\n"
                    "Potential SELL signal detected\n" + "-" * 50
                )
                print(f"{Fore.CYAN}{signal_msg}{Style.RESET_ALL}")
                logger.info(signal_msg)

            return df

        except Exception as e:
            error_msg = (
                "\n" + "!" * 50 + f"\nSignal calculation error: {str(e)}\n" + "!" * 50
            )
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
            logger.error(error_msg)
            return df

    def execute_trade(self, signal: str) -> None:
        try:
            if EXCHANGE == "BINANCE":
                current_price = float(
                    self.client.ticker_price(symbol=self.symbol)["price"]
                )
            else:  # BITSO
                current_price = float(self.client.ticker(self.symbol)["last"])

            trade_msg = (
                "\n"
                + "-" * 50
                + f"\nExecuting {signal} at {current_price} {self.state.currency}\n"
                + "-" * 50
            )
            print(f"{Fore.CYAN}{trade_msg}{Style.RESET_ALL}")
            logger.info(trade_msg)

            if self.state.is_simulation:
                self._execute_simulation_trade(signal, current_price)
            else:
                self._execute_real_trade(signal, current_price)

        except Exception as e:
            error_msg = (
                "\n" + "!" * 50 + f"\nTrade execution error: {str(e)}\n" + "!" * 50
            )
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
            logger.error(error_msg)

    def _execute_simulation_trade(self, signal: str, current_price: float) -> None:
        # Check if we need to reset monthly tracking
        current_month = datetime.now().month
        if current_month != self.state.last_month:
            self.state.month_start_balance = self.state.saldo_money + (
                self.state.monedas * current_price
            )
            self.state.monthly_profit = 0
            self.state.last_month = current_month

        if signal == "BUY" and self.state.saldo_money > 0:
            # Only invest if monthly target not yet reached and respect max buy limit
            if (
                self.state.monthly_profit
                < MONTHLY_TARGET * self.state.month_start_balance
            ):
                invest_amount = min(self.state.saldo_money, self.max_buy)
                amount = invest_amount / current_price
                self.state.monedas = amount
                self.state.saldo_money -= invest_amount
                self._record_trade("BUY", current_price, amount)
            else:
                skip_msg = (
                    "\n"
                    + "-" * 50
                    + "\nMonthly target reached - skipping buy\n"
                    + "-" * 50
                )
                print(f"{Fore.YELLOW}{skip_msg}{Style.RESET_ALL}")
                logger.info(skip_msg)

        elif signal == "SELL" and self.state.monedas > 0:
            amount = min(self.state.monedas, self.max_sell / current_price)
            sale_value = amount * current_price
            profit_percentage = (sale_value - (amount * self.state.last_price)) / (
                amount * self.state.last_price
            )

            # Sell if we hit take profit or stop loss
            should_sell = (profit_percentage >= TAKE_PROFIT) or (
                profit_percentage <= STOP_LOSS
            )

            if should_sell:
                profit = sale_value - (amount * self.state.last_price)
                self.state.saldo_money = sale_value
                self.state.monedas = 0
                self.state.total_trades += 1
                self.state.monthly_profit += profit

                if profit > 0:
                    self.state.total_profit += profit
                    self.state.winning_trades += 1
                else:
                    self.state.total_losses += abs(profit)
                    self.state.losing_trades += 1

                self._record_trade("SELL", current_price, amount, profit)
            else:
                hold_msg = (
                    "\n"
                    + "-" * 50
                    + f"\nHold position - Current P/L: {profit_percentage:.2%}\n"
                    + "-" * 50
                )
                print(f"{Fore.YELLOW}{hold_msg}{Style.RESET_ALL}")
                logger.info(hold_msg)

    def _execute_real_trade(self, signal: str, current_price: float) -> None:
        if signal == "BUY":
            quantity = self.calculate_quantity(current_price)
            order = self._place_order(SIDE_BUY, quantity)
            order_msg = "\n" + "-" * 50 + f"\nOrder placed: {order}\n" + "-" * 50
            print(f"{Fore.GREEN}{order_msg}{Style.RESET_ALL}")
            logger.info(order_msg)

        elif signal == "SELL":
            order = self._place_order(SIDE_SELL, self.state.monedas)
            order_msg = "\n" + "-" * 50 + f"\nOrder placed: {order}\n" + "-" * 50
            print(f"{Fore.GREEN}{order_msg}{Style.RESET_ALL}")
            logger.info(order_msg)

    def _record_trade(
        self, trade_type: str, price: float, amount: float, profit: float = 0
    ) -> None:
        if trade_type == "BUY":
            self.state.last_price = price
            trade_msg = (
                "\n" + "=" * 50 + "\n"
                "BUY EXECUTION:\n"
                f"Amount: {amount:.8f} BTC\n"
                f"Price: {price:.2f} {self.state.currency}\n"
                f"Total Cost: {(amount * price):.2f} {self.state.currency}\n" + "=" * 50
            )
            print(f"{Fore.YELLOW}{trade_msg}{Style.RESET_ALL}")
            logger.info(trade_msg)
        else:
            win_rate = (
                (self.state.winning_trades / self.state.total_trades * 100)
                if self.state.total_trades > 0
                else 0
            )
            color = Fore.GREEN if profit > 0 else Fore.RED
            trade_msg = (
                "\n" + "=" * 50 + "\n"
                "SELL EXECUTION:\n"
                f"Amount: {amount:.8f} BTC\n"
                f"Price: {price:.2f} {self.state.currency}\n"
                f"Profit/Loss: {profit:.2f} {self.state.currency}\n"
                f"Monthly Profit: {(self.state.monthly_profit / self.state.month_start_balance):.2%}\n"
                f"Win Rate: {win_rate:.1f}%\n"
                f"Total P/L: {(self.state.total_profit - self.state.total_losses):.2f} {self.state.currency}\n"
                + "=" * 50
            )
            print(f"{color}{trade_msg}{Style.RESET_ALL}")
            logger.info(trade_msg)

    def _place_order(self, side: str, quantity: float) -> Dict:
        if EXCHANGE == "BINANCE":
            return self.client.create_order(
                symbol=self.symbol, side=side, type=ORDER_TYPE_MARKET, quantity=quantity
            )
        else:  # BITSO
            return self.client.place_order(
                book=self.symbol, side=side.lower(), order_type="market", major=quantity
            )

    def calculate_quantity(self, price: float) -> float:
        if EXCHANGE == "BINANCE":
            balance = float(self.client.get_asset_balance(asset="USDT")["free"])
            quantity = min(balance, MAX_BUY_USD) * FEE_RATE / price
        else:  # BITSO
            balance = float(self.client.balance()["mxn"]["available"])
            quantity = min(balance, MAX_BUY_MXN) * FEE_RATE / price
        return round(quantity, 5)

    def run_bot(self) -> None:
        while True:
            df = self.get_historical_data()

            if not df.empty:
                df = self.calculate_signals(df)
                self._check_signals(df)
                self._print_portfolio_summary(df)

            time.sleep(CHECK_INTERVAL)

    def _check_signals(self, df: pd.DataFrame) -> None:
        if pd.notna(df["Buy_Signal"].iloc[-1]):
            self.execute_trade("BUY")
        elif pd.notna(df["Sell_Signal"].iloc[-1]):
            self.execute_trade("SELL")

    def _print_portfolio_summary(self, df: pd.DataFrame) -> None:
        win_rate = (
            (self.state.winning_trades / self.state.total_trades * 100)
            if self.state.total_trades > 0
            else 0
        )
        net_pl = self.state.total_profit - self.state.total_losses
        color = Fore.GREEN if net_pl > 0 else Fore.RED

        summary = (
            "\n" + "*" * 50 + "\n"
            "PORTFOLIO SUMMARY:\n"
            f"{self.state.currency} Balance: {self.state.saldo_money:.2f}\n"
            f"BTC Balance: {self.state.monedas:.8f}\n"
            f"Current Price: {df['close'].iloc[-1]:.2f} {self.state.currency}\n"
            f"RSI: {df['RSI'].iloc[-1]:.2f}\n"
            f"SMA20: {df['SMA20'].iloc[-1]:.2f} {self.state.currency}\n"
            f"SMA50: {df['SMA50'].iloc[-1]:.2f} {self.state.currency}\n"
            f"Total Profit: {self.state.total_profit:.2f} {self.state.currency}\n"
            f"Total Losses: {self.state.total_losses:.2f} {self.state.currency}\n"
            f"Net P/L: {net_pl:.2f} {self.state.currency}\n"
            f"Monthly P/L: {(self.state.monthly_profit / self.state.month_start_balance):.2%}\n"
            f"Win Rate: {win_rate:.1f}%\n"
            f"Next check in {CHECK_INTERVAL/60} minutes...\n" + "*" * 50
        )

        print(f"{color}{summary}{Style.RESET_ALL}")
        logger.info(summary)


if __name__ == "__main__":
    bot = TradingBot()
    startup_msg = "\n" + "=" * 50 + "\nTrading bot starting...\n" + "=" * 50
    print(f"{Fore.CYAN}{startup_msg}{Style.RESET_ALL}")
    logger.info(startup_msg)
    bot.run_bot()
