"""Read-only vendor diagnostics; never remove unavailable securities."""

import argparse
from io import StringIO
from pathlib import Path

import akshare as ak
import pandas as pd
import requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="600837")
    parser.add_argument(
        "--source",
        choices=["shares", "tencent", "sina-shares", "sina-structure", "announcements"],
        default="shares",
    )
    args = parser.parse_args()
    if args.source == "announcements":
        frame = ak.stock_zh_a_disclosure_report_cninfo(
            symbol=args.symbol, start_date="20221215", end_date="20221220"
        )
    elif args.source == "shares":
        frame = ak.stock_share_change_cninfo(
            symbol=args.symbol, start_date="20180101", end_date="20231229"
        )
    elif args.source == "tencent":
        frame = ak.stock_zh_a_hist_tx(
            symbol="sh" + args.symbol, start_date="20220101", end_date="20231229", timeout=20
        )
    else:
        url = f"https://money.finance.sina.com.cn/corp/go.php/vCI_StockStructureHistory/stockid/{args.symbol}/stocktype/LiuTongA.phtml"
        if args.source == "sina-structure":
            url = f"https://money.finance.sina.com.cn/corp/go.php/vCI_StockStructure/stockid/{args.symbol}.phtml"
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        response.encoding = "gb2312"
        html = response.text
        Path(f"data/raw/ashare/probe_{args.symbol}_shares.html").write_text(html, encoding="utf-8")
        tables = pd.read_html(StringIO(html))
        print([(i, table.shape) for i, table in enumerate(tables)], flush=True)
        import re

        if args.source == "sina-structure":
            print(re.findall(r'href="([^"]*StockStructureHistory[^"]*)"', html), flush=True)
            return
        graph = re.search(r"<graph.*?</graph>", html, re.DOTALL)
        print(graph.group(0)[:2500] if graph else "no graph", flush=True)
        return
    print(frame.shape, frame.columns.tolist(), flush=True)
    print(frame.tail(3).to_string(index=False), flush=True)
    path = Path("data/raw/ashare") / f"probe_{args.symbol}_{args.source}.parquet"
    frame.to_parquet(path, index=False)


if __name__ == "__main__":
    main()
