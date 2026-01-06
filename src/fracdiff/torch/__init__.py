try:
    import torch  # noqa: F401
    from .functional import fdiff, fdiff_coef
    from .module import Fracdiff
except ImportError:
    def fdiff(*args, **kwargs):
        raise ImportError(
            "PyTorch is not installed. Please install it with "
            "`pip install 'fracdiff-modern[torch]'` to use this feature."
        )

    def fdiff_coef(*args, **kwargs):
        raise ImportError(
            "PyTorch is required for fdiff_coef."
            "Run: pip install 'fracdiff-modern[torch]'"
        )

    class Fracdiff:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for Fracdiff."
                "Run: pip install 'fracdiff-modern[torch]'"
            )

__all__ = ["fdiff", "fdiff_coef", "Fracdiff"]
