import pandas as pd


def create_dates(start, end):
    dates = pd.date_range(start, end, closed="right", freq="D")
    return [str(date.date()) for date in dates]


def load_tau_section_parameters(tau_section):
    """Load parameters for trading algorithm. Depending on tau section:

    Args:
        tau_section (str): small: 8-40, big: 41-99, huge: 100-182

    Returns:
        many: filename, near_bound, far_bound, h, tau_min, tau_max
    """
    if tau_section == "small":
        return "trades_smallTau.csv", 0.1, 0.3, 0.15, 7, 40
    elif tau_section == "big":
        return "trades_bigTau.csv", 0.15, 0.35, 0.25, 40, 99
    elif tau_section == "huge":
        return "trades_hugeTau.csv", 0.2, 0.4, 0.35, 99, 182
