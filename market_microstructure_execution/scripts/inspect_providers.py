"""Inspect installed provider mapping and HTTP controls without any credentials."""

import inspect

import akshare as ak
import easyquotation

print("AKShare", ak.__version__)
print(inspect.getsource(ak.stock_bid_ask_em))
print(inspect.getsource(easyquotation.use("sina").__class__))
print(inspect.getsource(easyquotation.use("sina").__class__.__bases__[0]))
