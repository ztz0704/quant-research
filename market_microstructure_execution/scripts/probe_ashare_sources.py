"""Probe real A-share provider libraries with bounded calls and persisted provenance."""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone


def worker(provider: str, symbols: list[str], bypass_proxy: bool) -> None:
    """Only the child process changes proxy settings or provider HTTP transport."""
    sys.stdout.reconfigure(encoding="utf-8")
    if bypass_proxy:
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            os.environ.pop(name, None)
        os.environ["NO_PROXY"] = "*"
    if provider == "sina":
        from easyquotation.sina import Sina

        class HttpsSina(Sina):
            """Use the library parser over HTTPS with a real HTTP timeout."""

            @property
            def stock_api(self):
                return "https://hq.sinajs.cn/list="

            def get_stocks_by_range(self, params):
                response = self._session.get(self.stock_api + params, headers=self._get_headers(), timeout=12)
                response.raise_for_status()
                return response.text

        quotes = HttpsSina().real(symbols, prefix=True)
        print(json.dumps({"quotes": quotes, "library": "easyquotation"}, ensure_ascii=False))
    elif provider == "akshare":
        import akshare as ak

        quotes = {symbol: ak.stock_bid_ask_em(symbol=symbol).to_dict("records") for symbol in symbols}
        print(json.dumps({"quotes": quotes, "library": "akshare", "version": ak.__version__}, ensure_ascii=False))
    else:
        raise ValueError("Unknown provider")


def main() -> None:
    """Archive successful books and all failed source attempts; never synthesize fallback data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=["sina", "akshare"])
    parser.add_argument("--symbols", nargs="+", default=["000001", "600000", "000333", "600519"])
    parser.add_argument("--bypass-proxy", action="store_true")
    parser.add_argument("--historical-sample", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.symbols, args.bypass_proxy)
        return
    import pandas as pd

    from market_microstructure.data.ashare import fetch_library_snapshot
    from market_microstructure.data.validation import validate_orderbook
    from market_microstructure.utils.paths import ensure_directories, project_path

    ensure_directories()
    audit = []
    for provider in ("sina", "akshare"):
        attempt = {"provider": provider, "symbols": args.symbols,
                   "retrieved_utc": datetime.now(timezone.utc).isoformat()}
        try:
            frame, raw = fetch_library_snapshot(provider, args.symbols, bypass_proxy=args.bypass_proxy)
            project_path(f"data/raw/ashare_{provider}_payload.json").write_text(
                json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
            if frame.empty:
                raise ValueError("Provider returned no snapshots")
            frame.to_parquet(project_path(f"data/raw/ashare_{provider}_snapshots.parquet"), index=False)
            clean, report = validate_orderbook(frame)
            clean.to_parquet(project_path(f"data/interim/ashare_{provider}_snapshots.parquet"), index=False)
            attempt.update(status="success", rows=len(clean), audit=report.to_dict(),
                           timestamp_quality=clean.timestamp_quality.iloc[0],
                           max_stale_seconds=float(clean.stale_seconds.max()),
                           research_ready=False, reason="Single cross-section is not a temporal research sample")
        except Exception as error:
            # Failures are explicit artifacts, never swallowed or replaced by generated quotes.
            attempt.update(status="failed", error_type=type(error).__name__, reason=str(error)[:700])
        audit.append(attempt)
        print(json.dumps(attempt, ensure_ascii=True))
    project_path("outputs/tables/ashare_source_probe.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    pd.DataFrame(audit).to_csv(project_path("outputs/tables/ashare_source_probe.csv"), index=False)
    if args.historical_sample:
        download_public_sample(args.bypass_proxy)


def download_public_sample(bypass_proxy: bool) -> None:
    """Fetch published SZSE sample messages, pin the upstream commit and audit hashes."""
    import pandas as pd
    import requests
    import yaml

    from market_microstructure.utils.paths import project_path

    session = requests.Session()
    session.trust_env = not bypass_proxy
    source = yaml.safe_load(project_path("configs/public_sample.yaml").read_text(encoding="utf-8"))
    commit = source["commit"]
    manifest = {"repository": f"https://github.com/{source['repository']}", "commit": commit,
                "classification": "published sample messages; verify date/schema before empirical use", "files": []}
    directory = project_path("data/raw/szse_public")
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in source["files"].items():
        url = f"https://raw.githubusercontent.com/{source['repository']}/{commit}/{source['upstream_directory']}/{name}"
        path = directory / name
        started = time.monotonic()
        digest = hashlib.sha256()
        size = 0
        if path.exists():
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256")
            size = path.stat().st_size
            if digest.hexdigest() != expected["sha256"]:
                raise ValueError(f"Existing {name} differs from pinned source; refusing to overwrite")
        else:
            temporary = path.with_suffix(".csv.partial")
            with session.get(url, stream=True, timeout=20) as download:
                download.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in download.iter_content(1024 * 1024):
                        if time.monotonic() - started > 180 or size > 200 * 1024 * 1024:
                            raise TimeoutError("Public sample exceeded bounded download budget")
                        handle.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
            if digest.hexdigest() != expected["sha256"]:
                raise ValueError(f"Downloaded {name} does not match the pinned SHA-256")
            temporary.replace(path)
        preview = pd.read_csv(path, nrows=3)
        record = {"file": name, "url": url, "bytes": size, "sha256": digest.hexdigest(),
                  "columns": list(preview.columns), "preview": json.loads(preview.to_json(orient="records"))}
        manifest["files"].append(record)
        print(json.dumps(record, ensure_ascii=True), flush=True)
        project_path("outputs/tables/public_sample_manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
