import os
import time
import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional
from colorama import init, Fore, Style
from dotenv import load_dotenv
from telethon import TelegramClient, events, sync
import streamlit as st
import requests

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
ORDER_TYPE_MARKET = st.secrets["trading_configuration"]["ORDER_TYPE_MARKET"]
SIDE_BUY = st.secrets["trading_configuration"]["SIDE_BUY"]
SIDE_SELL = st.secrets["trading_configuration"]["SIDE_SELL"]
INTERVAL = st.secrets["trading_configuration"]["INTERVAL"]
LIMIT = st.secrets["trading_configuration"]["LIMIT"]
RSI_PERIOD = st.secrets["trading_configuration"]["RSI_PERIOD"]
SMA_FAST = st.secrets["trading_configuration"]["SMA_FAST"]
SMA_SLOW = st.secrets["trading_configuration"]["SMA_SLOW"]
RSI_OVERBOUGHT = st.secrets["trading_configuration"]["RSI_OVERBOUGHT"]
RSI_OVERSOLD = st.secrets["trading_configuration"]["RSI_OVERSOLD"]
FEE_RATE = st.secrets["trading_configuration"]["FEE_RATE"]
CHECK_INTERVAL = st.secrets["trading_configuration"]["CHECK_INTERVAL"]
STOP_LOSS = st.secrets["trading_configuration"]["STOP_LOSS"]
TAKE_PROFIT = st.secrets["trading_configuration"]["TAKE_PROFIT"]
MONTHLY_TARGET = st.secrets["trading_configuration"]["MONTHLY_TARGET"]
# Additional configuration for Telegram notifications
TELEGRAM_TOKEN = st.secrets["telegram_keys"]["TOKEN"]
TELEGRAM_CHAT_ID = st.secrets["telegram_keys"]["CHAT_ID"]


class State:
    def __init__(self):
        self.saldo_money: float = float(
            st.secrets["initial_balance"]["INITIAL_BALANCE"]
        )
        self.monedas: float = 0.0
        self.is_simulation: bool = bool(st.secrets["simulation_mode"]["IS_SIMULATION"])
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
        self.currency = "USD"


class TradingBot:
    def __init__(self):
        self.state = State()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.symbol = "BTC-USD"

    def send_message(self, message: str) -> None:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        try:
            response = requests.post(url, data=data)
            if response.status_code == 200:
                logger.info(f"Message sent successfully: {message}")
            else:
                logger.error(f"Failed to send message: {response.text}")
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")

    def get_historical_data(self) -> pd.DataFrame:
        try:
            df = yf.download(tickers=self.symbol, period="7d", interval=INTERVAL)
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
                    executor.submit(lambda: df["Close"].rolling(window=SMA_FAST).mean())
                )
                futures.append(
                    executor.submit(lambda: df["Close"].rolling(window=SMA_SLOW).mean())
                )
                futures.append(executor.submit(lambda: self.calculate_rsi(df["Close"])))

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

            df.loc[buy_conditions, "Buy_Signal"] = df["Close"]
            df.loc[sell_conditions, "Sell_Signal"] = df["Close"]

            # Log signal analysis with quantity recommendation
            if buy_conditions.iloc[-1]:
                signal_msg = (
                    "\n" + "-" * 50 + "\n"
                    "SIGNAL ANALYSIS:\n"
                    "SMA20 crossed above SMA50 (Bullish)\n"
                    f"RSI: {df['RSI'].iloc[-1]:.2f} (Not overbought)\n"
                    "Potential BUY signal detected\n"
                    f"Recommended quantity: {self.calculate_buy_quantity(df['Close'].iloc[-1]):.2f}\n"
                    + "-" * 50
                )
                print(f"{Fore.CYAN}{signal_msg}{Style.RESET_ALL}")
                logger.info(signal_msg)
                self.send_message(
                    f"Good time to buy Bitcoin. Recommended quantity: {self.calculate_buy_quantity(df['Close'].iloc[-1]):.2f}"
                )

            elif sell_conditions.iloc[-1]:
                signal_msg = (
                    "\n" + "-" * 50 + "\n"
                    "SIGNAL ANALYSIS:\n"
                    "SMA20 crossed below SMA50 (Bearish)\n"
                    f"RSI: {df['RSI'].iloc[-1]:.2f} (Not oversold)\n"
                    "Potential SELL signal detected\n"
                    f"Recommended quantity: {self.calculate_sell_quantity(df['Close'].iloc[-1]):.2f}\n"
                    + "-" * 50
                )
                print(f"{Fore.CYAN}{signal_msg}{Style.RESET_ALL}")
                logger.info(signal_msg)
                self.send_message(
                    f"Good time to sell Bitcoin. Recommended quantity: {self.calculate_sell_quantity(df['Close'].iloc[-1]):.2f}"
                )

            return df

        except Exception as e:
            error_msg = (
                "\n" + "!" * 50 + f"\nSignal calculation error: {str(e)}\n" + "!" * 50
            )
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
            logger.error(error_msg)
            return df

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
            f"Current Price: {df['Close'].iloc[-1]:.2f} {self.state.currency}\n"
            f"RSI: {df['RSI'].iloc[-1]:.2f}\n"
            f"Net P/L: {net_pl:.2f} {self.state.currency}\n"
            f"Next check in {CHECK_INTERVAL/60} minutes...\n" + "*" * 50
        )

        print(f"{color}{summary}{Style.RESET_ALL}")
        logger.info(summary)

    def calculate_buy_quantity(self, price: float) -> float:
        # Assuming a fixed percentage of the available balance for each trade
        trade_percentage = 0.05  # 5% of the balance for each trade
        quantity = self.state.saldo_money * trade_percentage / price
        return quantity

    def calculate_sell_quantity(self, price: float) -> float:
        # Assuming a fixed percentage of the available balance for each trade
        trade_percentage = 0.05  # 5% of the balance for each trade
        quantity = self.state.monedas * trade_percentage
        return quantity


if __name__ == "__main__":
    bot = TradingBot()
    startup_msg = "\n" + "=" * 50 + "\nTrading bot starting...\n" + "=" * 50
    print(f"{Fore.CYAN}{startup_msg}{Style.RESET_ALL}")
    logger.info(startup_msg)
    bot.run_bot()
