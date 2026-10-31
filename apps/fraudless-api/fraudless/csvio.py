"""Untrusted CSV bytes -> all-string DataFrame, with hard limits and explicit errors."""
import csv
import io

import pandas as pd

from .validate import FileValidationError

ALLOWED_CONTENT_TYPES = {"text/csv", "application/csv", "application/vnd.ms-excel", "text/plain", "application/octet-stream", ""}


def check_upload_type(filename: str | None, content_type: str | None) -> None:
    name = (filename or "").lower()
    ctype = (content_type or "").split(";")[0].strip().lower()
    if not name.endswith(".csv") or ctype not in ALLOWED_CONTENT_TYPES:
        raise FileValidationError("unsupported_file_type", "Only .csv uploads are supported", {"filename_extension": name.rsplit(".", 1)[-1] if "." in name else ""}, status=415)


def parse_csv(data: bytes, max_rows: int) -> pd.DataFrame:
    if not data.strip():
        raise FileValidationError("empty_file", "The file is empty", status=400)
    if b"\x00" in data:
        raise FileValidationError("not_text", "The file looks binary, not CSV text", status=400)
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise FileValidationError("invalid_encoding", "The file must be UTF-8 encoded", status=400)

    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        header = next(reader)
        if len(header) == 1 and any(d in header[0] for d in (";", "\t", "|")):
            raise FileValidationError("unsupported_delimiter", "Use comma-separated values", status=400)
        rows: list[list[str]] = []
        for line_no, row in enumerate(reader, start=2):
            if not row:  # blank line
                continue
            if len(row) != len(header):
                raise FileValidationError(
                    "malformed_csv", f"Line {line_no} has {len(row)} fields; the header has {len(header)}", {"line": line_no}, status=400
                )
            rows.append(row)
            if len(rows) > max_rows:
                raise FileValidationError("too_many_rows", f"The file exceeds the {max_rows} row limit", {"max_rows": max_rows}, status=413)
    except csv.Error as e:
        raise FileValidationError("malformed_csv", f"The file is not valid CSV: {e}", status=400)
    if not rows:
        raise FileValidationError("no_data_rows", "The file has a header but no data rows", status=400)
    return pd.DataFrame(rows, columns=header, dtype="string")
