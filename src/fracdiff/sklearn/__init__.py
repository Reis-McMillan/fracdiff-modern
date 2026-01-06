try:
    import sklearn  # noqa: F401
    from .fracdiff import Fracdiff
    from .fracdiffstat import FracdiffStat
except ImportError:
    # Fallback placeholders that explain how to fix the error
    class Fracdiff:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Scikit-Learn is not installed. Please install it with "
                "`pip install 'fracdiff-modern[sklearn]'` to use this feature."
            )

    class FracdiffStat:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Scikit-Learn is not installed. Please install it with "
                "`pip install 'fracdiff-modern[sklearn]'` to use this feature."
            )

__all__ = ["Fracdiff", "FracdiffStat"]
