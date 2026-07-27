"""Shared pipeline-control primitives."""


class PipelineCancelled(BaseException):
    """Raised when a running pipeline is interrupted by user request."""
