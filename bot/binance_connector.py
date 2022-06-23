from dataclasses import dataclass
from binance.spot import Spot as Client
from bot.load_env import API_KEY, SECRET_KEY
import logging
from utils.log import prepare_logger
import numpy as np

logger = prepare_logger(logging.INFO)


@dataclass
class BinanceBalance:
    btc: float
    usdt: float
    total_balance: float

    @classmethod
    def get_total_balance(
        cls,
        btc,
        usdt,
        exchange_rate,
    ):
        return cls(btc=btc, usdt=usdt, total_balance=btc * exchange_rate + usdt)


class BinanceConnector:
    def __init__(
        self, base_url="https://testnet.binance.vision", tickers=["BTC", "USDT"]
    ) -> None:
        self.client = Client(base_url=base_url, key=API_KEY, secret=SECRET_KEY)
        self.tickers = tickers
        self.balance = None
        self.get_balance()

    @staticmethod
    def find_currencies(currencies, balances) -> dict:
        return {
            currency_dict["asset"].lower(): float(currency_dict["free"])
            for currency_dict in balances
            if currency_dict["asset"] in currencies
        }

    def get_balance(self, verbose=True) -> None:
        balances = self.client.account()["balances"]
        exchange_rate = self.get_exchange_rate()
        self.balance = BinanceBalance.get_total_balance(
            **self.find_currencies(currencies=self.tickers, balances=balances),
            exchange_rate=exchange_rate,
        )
        if verbose:
            logger.info(
                f"Total balance: {self.get_string_from_dict(self.balance.__dict__)}"
            )

    def get_exchange_rate(self, ticker="BTCUSDT") -> float:
        return float(self.client.ticker_price(ticker)["price"])

    @staticmethod
    def get_string_from_dict(dictionary) -> str:
        string_list = [f"{k}: {v}" for k, v in dictionary.items()]
        return " - ".join(string_list)

    def place_order(self, order_type, quantity) -> None:
        possible_order_types = ["SELL", "BUY"]
        assert (
            order_type in possible_order_types
        ), f"Wrong order type - {order_type}! It must be in {possible_order_types}"
        params = {
            "symbol": "BTCUSDT",
            "side": order_type,
            "type": "MARKET",
            "quantity": quantity,
        }
        logger.info(f"Placing order - params: {self.get_string_from_dict(params)}")
        self.client.new_order(**params)

    @staticmethod
    def round_quantity(quantity, ndigits=6):
        return float(round(quantity, ndigits))

    def sell(self) -> None:
        self.get_balance()
        quantity = self.round_quantity(self.balance.btc)
        if quantity >= 0.0005:
            self.place_order(order_type="SELL", quantity=quantity)
            return True, quantity
        else:
            logger.info(f"Not enough quantity to sell - {quantity}!")
            return False, None  

    def buy(self) -> None:
        self.get_balance()
        exchange_rate = self.get_exchange_rate()
        quantity = self.round_quantity(self.balance.usdt * 0.90 / exchange_rate)
        if quantity >= 0.0005:
            self.place_order(order_type="BUY", quantity=quantity)
            return True, quantity
        else:
            logger.info(f"Not enough quantity to buy! - {quantity}")
            return False, None 