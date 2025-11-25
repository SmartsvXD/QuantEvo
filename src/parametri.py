import json
import os

_lambdaEWMA = None


def ottieniLambdaEWMA(titoli):
    global _lambdaEWMA
    # Initialize the cache as a dict and try to load saved values from the project
    if _lambdaEWMA is None:
        _lambdaEWMA = {}
        path = "./data/lambdaEWMA.json"
        if os.path.exists(path):
            with open(path, "r") as file:
                try:
                    data = json.load(file)
                    if isinstance(data, dict):
                        _lambdaEWMA = data
                except Exception:
                    _lambdaEWMA = {}

    # Ensure every requested titolo has a value (default 0.9)
    for titolo in titoli:
        if titolo not in _lambdaEWMA:
            _lambdaEWMA[titolo] = 0.97

    return [_lambdaEWMA.get(k, 0.9) for k in titoli]
