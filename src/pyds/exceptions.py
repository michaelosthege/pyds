class NotProbabilisticError(Exception):
    """Indicates that a mass function is missspecified."""


class TotalConflictError(Exception):
    """K = 1.0: two BPAs are completely contradictory.

    Dempster's rule is undefined at K = 1.0.
    """
