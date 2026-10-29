"""Print only schema/type/date summaries of the public sample."""

import pandas as pd

from market_microstructure.utils.paths import project_path

for name in ("orders.csv", "trans.csv"):
    data = pd.read_csv(project_path(f"data/raw/szse_public/{name}"), low_memory=False)
    print(name, "rows", len(data))
    print(data.SecurityID.value_counts().to_string())
    types = [c for c in ("OrderType", "OrderBSFlag", "TradeType", "TradeBSFlag", "MDDate", "ChannelNo") if c in data]
    for column in types:
        print(column, data[column].value_counts(dropna=False).to_dict())
    if "TradeType" in data:
        print(data.groupby("TradeType").agg(min_price=("TradePrice", "min"), max_price=("TradePrice", "max"),
                                             buy_zero=("TradeBuyNo", lambda x: int((x == 0).sum())),
                                             sell_zero=("TradeSellNo", lambda x: int((x == 0).sum()))).to_string())
