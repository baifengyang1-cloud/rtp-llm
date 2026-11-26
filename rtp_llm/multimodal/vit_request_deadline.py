"""Carry a live owner deadline across preprocessing and scheduler admission."""

from contextlib import contextmanager
from contextvars import ContextVar

_deadline = ContextVar("vit_request_deadline", default=None)


def current_vit_deadline():
    return _deadline.get()


@contextmanager
def use_vit_deadline(deadline):
    token = _deadline.set(deadline)
    try:
        yield
    finally:
        _deadline.reset(token)
