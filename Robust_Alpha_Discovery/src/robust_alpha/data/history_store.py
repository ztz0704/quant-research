"""Serial rate-limited, resumable API partitions with immutable SDK response archives."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from robust_alpha.data.history_api import redact
from robust_alpha.data.research_contract import digest_file, identity


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


class RequestFailed(RuntimeError):
    pass


class Archive:
    def __init__(self, root: Path, *, direct=False, pause=1.0, retries=3, timeout=90):
        self.root = root.resolve()
        self.direct, self.pause, self.retries, self.timeout = direct, pause, retries, timeout
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "checkpoint.json"
        self.state = (
            json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state_path.exists()
            else {}
        )
        self.last_request = 0.0
        self.worker_dir = self.root / "work"
        self.worker_dir.mkdir(exist_ok=True)

    def record(self, key: str, info: dict) -> None:
        self.state[key] = info
        with (self.root / "requests.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"key": key, **info}, ensure_ascii=False) + "\n")
        write_json(self.state_path, self.state)

    def get(self, source: str, endpoint: str, **params) -> pd.DataFrame:
        descriptor = {"source": source, "endpoint": endpoint, "params": params, "archive_schema": 1}
        key = identity(descriptor)
        previous = self.state.get(key)
        if previous and previous["status"] == "complete":
            for file_kind in ("raw", "parquet"):
                if digest_file(self.root / previous[file_kind]) != previous[file_kind + "_sha256"]:
                    raise ValueError(f"Archive tampered: {file_kind} partition {key}")
            self.last_info = previous
            return pd.read_parquet(self.root / previous["parquet"])
        for attempt in range(1, self.retries + 1):
            time.sleep(max(0.0, self.pause - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            request_path, response_path = (
                self.worker_dir / f"{key}.{kind}.json" for kind in ("request", "response")
            )
            write_json(request_path, {**descriptor, "direct": self.direct})
            response_path.unlink(missing_ok=True)
            try:
                process = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "robust_alpha.data.history_api",
                        str(request_path),
                        str(response_path),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=self.timeout,
                )
                if process.returncode:
                    raise RequestFailed(f"Worker exited {process.returncode}")
                response = json.loads(response_path.read_text(encoding="utf-8"))
            except (subprocess.TimeoutExpired, RequestFailed) as exc:
                response = {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": redact(str(exc)),
                    "downloaded_at": datetime.now(UTC).isoformat(),
                }
                write_json(response_path, response)
            sha = digest_file(response_path)
            raw_path = self.root / "raw" / key / f"{sha}.json"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            if not raw_path.exists():
                with raw_path.open("xb") as stream:
                    stream.write(response_path.read_bytes())
            info = {
                **descriptor,
                "attempt": attempt,
                "downloaded_at": response["downloaded_at"],
                "raw": str(raw_path.relative_to(self.root)),
                "raw_sha256": sha,
            }
            if response["status"] == "ok":
                frame = pd.DataFrame(response["data"], columns=response["columns"])
                parquet_path = self.root / "partitions" / key / f"{sha}.parquet"
                parquet_path.parent.mkdir(parents=True, exist_ok=True)
                if not parquet_path.exists():
                    frame.to_parquet(parquet_path, index=False)
                self.record(
                    key,
                    {
                        **info,
                        "status": "complete",
                        "rows": len(frame),
                        "sdk_version": response["sdk_version"],
                        "parquet": str(parquet_path.relative_to(self.root)),
                        "parquet_sha256": digest_file(parquet_path),
                    },
                )
                self.last_info = self.state[key]
                return frame
            self.record(
                key,
                {
                    **info,
                    "status": "failed",
                    "error_type": response["error_type"],
                    "error": response["error"],
                },
            )
            if attempt < self.retries:
                time.sleep(2 ** (attempt - 1))
        raise RequestFailed(f"{source}/{endpoint}: {response['error_type']}: {response['error']}")
