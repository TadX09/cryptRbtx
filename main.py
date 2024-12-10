import os
import time
import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional
from colorama import init, Fore, Style
from dotenv import load_dotenv
from telethon import TelegramClient, events, sync
import streamlit as st
import requests
import schedule
import threading

# Load environment variables from .env file
load_dotenv()

# Initialize colorama for colored console output
init()

# Constants
PERIOD = st.secrets["trading_configuration"]["PERIOD"]
INTERVAL = st.secrets["trading_configuration"]["INTERVAL"]
INVERVAL_JOB = st.secrets["trading_configuration"]["INVERVAL_JOB"]
RSI_PERIOD = st.secrets["trading_configuration"]["RSI_PERIOD"]
SMA_FAST = st.secrets["trading_configuration"]["SMA_FAST"]
SMA_SLOW = st.secrets["trading_configuration"]["SMA_SLOW"]
RSI_OVERBOUGHT = st.secrets["trading_configuration"]["RSI_OVERBOUGHT"]
RSI_OVERSOLD = st.secrets["trading_configuration"]["RSI_OVERSOLD"]
# Additional configuration for Telegram notifications
TELEGRAM_TOKEN = st.secrets["telegram_keys"]["TOKEN"]
TELEGRAM_CHAT_ID = st.secrets["telegram_keys"]["CHAT_ID"]


class State:
    def __init__(self):
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
                print(f"Message sent successfully: {message}")
            else:
                print(f"Failed to send message: {response.text}")
        except Exception as e:
            print(f"Error sending message: {str(e)}")

    def get_historical_data(self) -> pd.DataFrame:
        try:
            df = yf.download(tickers=self.symbol, period=PERIOD, interval=INTERVAL)
            return df

        except Exception as e:
            error_msg = (
                "\n" + "!" * 50 + f"\nHistorical data error: {str(e)}\n" + "!" * 50
            )
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
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
                self.send_message(
                    f"Good time to buy Bitcoin. Price: ${df['Close'].iloc[-1]:.2f} (MXN: ${df['Close'].iloc[-1] * 20:.2f})"
                )

            elif sell_conditions.iloc[-1]:
                signal_msg = (
                    "\n" + "-" * 50 + "\n"
                    "SIGNAL ANALYSIS:\n"
                    "SMA20 crossed below SMA50 (Bearish)\n"
                    f"RSI: {df['RSI'].iloc[-1]:.2f} (Not oversold)\n"
                    "Potential SELL signal detected\n" + "-" * 50
                )
                print(f"{Fore.CYAN}{signal_msg}{Style.RESET_ALL}")
                self.send_message(
                    f"Good time to sell Bitcoin. Price: ${df['Close'].iloc[-1]:.2f} (MXN: ${df['Close'].iloc[-1] * 20:.2f})"
                )

            return df

        except Exception as e:
            error_msg = (
                "\n" + "!" * 50 + f"\nSignal calculation error: {str(e)}\n" + "!" * 50
            )
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
            return df

    def run_bot(self) -> None:
        def job():
            df = self.get_historical_data()

            if not df.empty:
                df = self.calculate_signals(df)
                self._check_signals(df)
                self._print_portfolio_summary(df)

        job()  # Run the job immediately
        schedule.every(INVERVAL_JOB).minutes.do(job)

        while True:
            schedule.run_pending()
            time.sleep(1)

    def _check_signals(self, df: pd.DataFrame) -> None:
        if pd.notna(df["Buy_Signal"].iloc[-1]):
            print("BUY signal detected")
        elif pd.notna(df["Sell_Signal"].iloc[-1]):
            print("SELL signal detected")

    def _print_portfolio_summary(self, df: pd.DataFrame) -> None:

        color = Fore.LIGHTBLACK_EX

        summary = (
            "\n" + "*" * 50 + "\n"
            "PORTFOLIO SUMMARY:\n"
            f"Date: {datetime.now(pytz.timezone('America/Mexico_City')).strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Current Price: {df['Close'].iloc[-1]:.2f} {self.state.currency}\n"
            f"Price in MXN: ${df['Close'].iloc[-1] * 20:.2f} MXN\n"
            f"RSI: {df['RSI'].iloc[-1]:.2f}\n"
            f"Next check in {INVERVAL_JOB} minutes...\n" + "*" * 50
        )

        print(f"{color}{summary}{Style.RESET_ALL}")


if __name__ == "__main__":
    bot = TradingBot()
    startup_msg = "\n" + "=" * 50 + "\nTrading bot starting...\n" + "=" * 50
    print(f"{Fore.CYAN}{startup_msg}{Style.RESET_ALL}")
    bot.run_bot()
