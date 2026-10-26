# SPDX-FileCopyrightText: 2026 Simone Vinciguerra
# SPDX-License-Identifier: GPL-3.0-or-later

class HistoricalDataException(Exception):
    """Exception raised when historical price data cannot be downloaded or validated."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)