from __future__ import annotations as _annotations

import asyncio
import types
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from logfire_api import LogfireSpan
from typing_extensions import ParamSpec, TypeAlias, TypeIs, get_args, get_origin
from typing_inspection import typing_objects
from typing_inspection.introspection import is_union_origin

if TYPE_CHECKING:
    from opentelemetry.trace import Span


AbstractSpan: TypeAlias = 'LogfireSpan | Span'

try:
    from opentelemetry.trace import Span, set_span_in_context
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    TRACEPARENT_PROPAGATOR = TraceContextTextMapPropagator()
    TRACEPARENT_NAME = 'traceparent'
    assert TRACEPARENT_NAME in TRACEPARENT_PROPAGATOR.fields

    # Logic taken from logfire.experimental.annotations
    def get_traceparent(span: AbstractSpan) -> str | None:
        """Get a string representing the span context to use for annotating spans."""
        real_span: Span
        if isinstance(span, Span):
            real_span = span
        else:
            real_span = span._span
            assert real_span
        context = set_span_in_context(real_span)
        carrier: dict[str, Any] = {}
        TRACEPARENT_PROPAGATOR.inject(carrier, context)
        return carrier.get(TRACEPARENT_NAME, '')

except ImportError:  # pragma: no cover

    def get_traceparent(span: AbstractSpan) -> str | None:
        # Opentelemetry wasn't installed, so we can't get the traceparent
        return None


def get_event_loop():
    try:
        event_loop = asyncio.get_event_loop()
    except RuntimeError:
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
    return event_loop


def get_union_args(tp: Any) -> tuple[Any, ...]:
    """Extract the arguments of a Union type if `response_type` is a union, otherwise return an empty tuple."""
    # similar to `pydantic_ai_slim/pydantic_ai/_result.py:get_union_args`
    if typing_objects.is_typealiastype(tp):
        tp = tp.__value__  # pragma: no cover

    origin = get_origin(tp)
    if is_union_origin(origin):
        return get_args(tp)
    else:
        return (tp,)


def unpack_annotated(tp: Any) -> tuple[Any, list[Any]]:
    """Strip `Annotated` from the type if present.

    Returns:
        `(tp argument, ())` if not annotated, otherwise `(stripped type, annotations)`.
    """
    origin = get_origin(tp)
    if typing_objects.is_annotated(origin):
        inner_tp, *args = get_args(tp)
        return inner_tp, args
    else:
        return tp, []


def comma_and(items: list[str]) -> str:
    """Join with a comma and 'and' for the last item."""
    if len(items) == 1:
        return items[0]
    else:
        # oxford comma ¯\_(ツ)_/¯
        return ', '.join(items[:-1]) + ', and ' + items[-1]


def get_parent_namespace(frame: types.FrameType | None) -> dict[str, Any] | None:
    """Attempt to get the namespace where the graph was defined.

    If the graph is defined with generics `Graph[a, b]` then another frame is inserted, and we have to skip that
    to get the correct namespace.
    """
    if frame is not None:  # pragma: no branch
        if back := frame.f_back:  # pragma: no branch
            if back.f_globals.get('__name__') == 'typing':
                # If the class calling this function is generic, explicitly parameterizing the class
                # results in a `typing._GenericAlias` instance, which proxies instantiation calls to the
                # "real" class and thus adding an extra frame to the call. To avoid pulling anything
                # from the `typing` module, use the correct frame (the one before):
                return get_parent_namespace(back)
            else:
                return back.f_locals


class Unset:
    """A singleton to represent an unset value.

    Copied from pydantic_ai/_utils.py.
    """

    pass


UNSET = Unset()
T = TypeVar('T')


def is_set(t_or_unset: T | Unset) -> TypeIs[T]:
    return t_or_unset is not UNSET


_P = ParamSpec('_P')
_R = TypeVar('_R')


async def run_in_executor(func: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs) -> _R:
    if kwargs:
        # noinspection PyTypeChecker
        return await asyncio.get_running_loop().run_in_executor(None, partial(func, *args, **kwargs))
    else:
        return await asyncio.get_running_loop().run_in_executor(None, func, *args)  # type: ignore
