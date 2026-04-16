"""Exceptions shared across mapping-agent HITL flows."""


class HITLBlockingError(Exception):
    """Raised when unresolved HITL items block pipeline progression."""
