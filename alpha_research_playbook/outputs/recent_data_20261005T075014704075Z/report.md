# 近期真实行情接口核验

核验截止：2026-09-30；价格：未复权。

通过代表本次指定样本的接口检查通过；不代表股票池/PIT/交易部署已完成。

```json
{
  "retrieved_at_utc": "2026-10-05T07:50:14.704075+00:00",
  "expected_last_session": "2026-09-30",
  "start_date": "2026-08-31",
  "adjustment": "none",
  "volume_unit": "shares",
  "amount_unit": "CNY",
  "akshare_version": "1.19.1",
  "script_sha256": "e94d3b296bad0cb3e676b50a127bf13db06a5df554c09341cb9b20c8dbf4d9d4",
  "results": [
    {
      "provider": "tencent",
      "asset": "000001",
      "passed": false,
      "rows": 22,
      "last_date": "2026-09-30",
      "reasons": [
        "amount_volume_implied_price_outside_bar_check_units"
      ],
      "files": {
        "tencent_000001.parquet": "d605ca1d713b9d52759a6cc489b438bb2a7fde4362289083b4e5e432a4c6dfe9",
        "tencent_000001.vendor.parquet": "6b29c023327aa054f70c57a8904cc3b15c03671b8f0153f46073572d8f1503db"
      }
    },
    {
      "provider": "eastmoney",
      "asset": "000001",
      "passed": true,
      "rows": 22,
      "last_date": "2026-09-30",
      "reasons": [],
      "files": {
        "eastmoney_000001.parquet": "9769aada67f9433044bd8d3a80c5b596f649e7e9037b79ec34d40ad694da183e",
        "eastmoney_000001.vendor.parquet": "4c1966f6549e331e6c9c8dffea136a805734240809bb14048eb784b9e8102458"
      }
    },
    {
      "provider": "tencent",
      "asset": "600000",
      "passed": true,
      "rows": 22,
      "last_date": "2026-09-30",
      "reasons": [],
      "files": {
        "tencent_600000.parquet": "78b67cd6285326a5666130037f29d9cc61c6864b959530c1e45b94e7842c3362",
        "tencent_600000.vendor.parquet": "20e7cab4f96bea21093c4e616dbcd844b089d9759d35804b559a3c4e74608b1a"
      }
    },
    {
      "provider": "eastmoney",
      "asset": "600000",
      "passed": false,
      "reasons": [
        "vendor_worker_failed"
      ],
      "error_type": "ProxyError",
      "files": {}
    }
  ],
  "comparisons": [
    {
      "asset": "000001",
      "shared_dates": 22,
      "same_dates": true,
      "max_ohlc_difference": 0.0,
      "status": "passed"
    },
    {
      "asset": "600000",
      "status": "unavailable_two_sources_required"
    }
  ],
  "all_passed": false,
  "limits": [
    "Small source probe, not full-universe coverage or PIT certification",
    "Expected last session is caller-confirmed, not inferred from weekdays",
    "Vendor requests may internally fetch more history; it is not evaluated",
    "No strategy selection, orders, brokerage access or synthetic fallback"
  ]
}
```
