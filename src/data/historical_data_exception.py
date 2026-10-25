class HistoricalDataException(Exception):
    """Exception raised when historical price data cannot be downloaded or validated."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)