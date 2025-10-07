# PCprophet/exceptions.py

from typing import Any, Iterable, Optional


class PCprophetError(Exception):
    """Base class for all PCprophet-specific errors."""
    code = "PCP-000"

    def __init__(self, message: str, *, path: Optional[str] = None, details: Any = None):
        self.path = path
        self.details = details
        super().__init__(self._format(message))

    def _format(self, msg: str) -> str:
        bits = [f"[{self.code}] {msg}"]
        if self.path:
            bits.append(f"(file: {self.path})")
        return " ".join(bits)

class NaRowError(PCprophetError):
    code = "PCP-101"
    def __init__(self, *, path: Optional[str] = None, indices: Optional[Iterable[int]] = None):
        super().__init__(
            "Row(s) contain NA values",
            path=path,
            details={"rows": list(indices) if indices is not None else None},
        )


class MissingColumnError(PCprophetError):
    code = "PCP-102"

    def __init__(self, missing: Iterable[str], *, path: Optional[str] = None):
        missing_list = list(missing)
        super().__init__(f"Missing required column(s): {', '.join(missing_list)}", path=path, details=missing_list)


class DuplicateRowError(PCprophetError):
    code = "PCP-103"
    def __init__(self, *, path: Optional[str] = None, rows: Optional[Iterable[int]] = None):
        super().__init__(
            "Duplicate row(s) detected",
            path=path,
            details={"rows": list(rows) if rows is not None else None},
        )

class EmptyColumnError(PCprophetError):
    code = "PCP-104"
    def __init__(self, column: str, *, path: Optional[str] = None, indices: Optional[Iterable[int]] = None):
        super().__init__(
            f"Column '{column}' has empty/NA values",
            path=path,
            details={"column": column, "rows": list(indices) if indices is not None else None},
        )

class DuplicateIdentifierError(PCprophetError):
    code = "PCP-105"

    def __init__(self, keys: Iterable[str], *, path: Optional[str] = None):
        keys_list = list(keys)
        super().__init__(f"Duplicate identifiers based on keys {keys_list}", path=path, details=keys_list)


class NaInMatrixError(PCprophetError):
    code = "PCP-106"

    def __init__(self, *, path: Optional[str] = None):
        super().__init__("NA values present in matrix", path=path)


class NotImplementedError(PCprophetError):
    code = "PCP-107"

    def __init__(self, *, path: Optional[str] = None):
        super().__init__("Not implemented Error", path=path)


class ConditionError(PCprophetError):
    code = "PCP-201"

    def __init__(self, detail: str, *, path: Optional[str] = None):
        super().__init__(f"Invalid 'cond' values: {detail}", path=path, details=detail)


# Optional custom “not implemented” that doesn’t shadow Python’s built-in
class PCprophetNotImplementedError(PCprophetError):
    code = "PCP-901"

    def __init__(self, feature: str, *, path: Optional[str] = None):
        super().__init__(f"Feature not implemented: {feature}", path=path, details=feature)


__all__ = [
    "PCprophetError",
    "NaRowError",
    "MissingColumnError",
    "DuplicateRowError",
    "EmptyColumnError",
    "DuplicateIdentifierError",
    "NaInMatrixError",
    "NotImplementedError",
    "ConditionError",
    "PCprophetNotImplementedError",
]
