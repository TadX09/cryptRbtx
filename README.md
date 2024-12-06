# cryptRbtx
 
## Project Description

This project is a trading bot that uses the Binance and Bitso APIs to trade cryptocurrencies. It is written in Python and uses various technical indicators to make trading decisions.

## Installation

To install the necessary dependencies, run the following command:

```
pip install -r requirements.txt
```

## Usage

To use the bot, you need to set up your environment variables in a .env file. Here is an example of the .env file:

```
EXCHANGE=BINANCE
MAX_BUY_USD=5
MAX_BUY_MXN=50
MAX_SELL_USD=25
MAX_SELL_MXN=100
INITIAL_BALANCE=100
IS_SIMULATION=True
BITSO_API_KEY=your_bitso_api_key
BITSO_API_SECRET=your_bitso_api_secret
ORDER_TYPE_MARKET=MARKET
SIDE_BUY=BUY
SIDE_SELL=SELL
INTERVAL=15m
LIMIT=100
RSI_PERIOD=14
SMA_FAST=20
SMA_SLOW=50
RSI_OVERBOUGHT=70
RSI_OVERSOLD=30
FEE_RATE=0.995
CHECK_INTERVAL=900
STOP_LOSS=2
TAKE_PROFIT=5
MONTHLY_TARGET=5
```

Once you have set up your .env file, you can run the bot with the following command:

```
python main.py
```

The bot will start trading according to the configuration in your .env file.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

