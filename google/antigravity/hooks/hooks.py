# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base definitions for Antigravity SDK Hooks v2.

This module defines the interface for Hooks and the standard result types
returned by their lifecycle callbacks.

Hooks can be defined as classes or by using decorators (e.g. `@pre_turn`).
Decorators automatically wrap functions into Hook instances. Wrapped functions
can optionally accept the hook's context:
- By naming the parameter `context`
  (e.g. `async def my_hook(context, data)`)
- By type-hinting it as `HookContext`
  (e.g. `async def my_hook(ctx: HookContext, data)`)
The context can be placed in any position (first or second).

"""

from __future__ import annotations

import abc
import functools
import inspect
from typing import Any, Generic, TypeVar

from google.antigravity import types
from google.antigravity.types import AskQuestionInteractionSpec
from google.antigravity.types import HookResult
from google.antigravity.types import QuestionHookResult
from google.antigravity.utils import state as state_module


# --- Contexts ---


class HookContext(state_module.StateStore):
  """Base context for hooks to share state."""

  def __init__(self, parent: HookContext | None = None):
    super().__init__(parent=parent)


class SessionContext(HookContext):
  """Context scoped to an entire session."""

  def __init__(self):
    super().__init__(parent=None)


class TurnContext(HookContext):
  """Context scoped to a single turn."""

  def __init__(self, session_context: SessionContext):
    super().__init__(parent=session_context)


class OperationContext(HookContext):
  """Context scoped to a specific operation (e.g. tool call)."""

  def __init__(self, turn_context: TurnContext):
    super().__init__(parent=turn_context)


# --- Base Hook Types ---


T = TypeVar("T")
R = TypeVar("R")


class InspectHook(abc.ABC, Generic[T]):
  """Read-only, non-blocking hook for observability."""

  @abc.abstractmethod
  async def run(self, context: HookContext, data: T) -> None:
    """Runs the inspection hook.

    Args:
      context: The hook context.
      data: The data to inspect (read-only).
    """
    pass


class DecideHook(abc.ABC, Generic[T]):
  """Read-only, blocking hook for policy decisions."""

  @abc.abstractmethod
  async def run(self, context: HookContext, data: T) -> HookResult:
    """Runs the decision hook.

    Args:
      context: The hook context.
      data: The data to make a decision on.

    Returns:
      A HookResult indicating allow/deny.
    """
    pass


class TransformHook(abc.ABC, Generic[T, R]):
  """Modifying, blocking hook for data transformation."""

  @abc.abstractmethod
  async def run(self, context: HookContext, data: T) -> R:
    """Runs the transformation hook.

    Args:
      context: The hook context.
      data: The data to transform.

    Returns:
      The transformed data.
    """
    pass


Hook = InspectHook | DecideHook | TransformHook


# --- Concrete Hook Interfaces ---


# Session
class OnSessionStartHook(InspectHook[None]):
  """Invoked when the session starts."""

  pass


class OnSessionEndHook(InspectHook[None]):
  """Invoked when the session ends."""

  pass


# Turn
class PreTurnHook(DecideHook[types.Content]):
  """Invoked before a turn starts.

  The `data` parameter receives the user's prompt (types.Content), which can
  be a single string/media object or a list of multimodal primitives.
  """

  pass


class PostTurnHook(InspectHook[str]):
  """Invoked after a turn ends.

  The `data` parameter receives the model's response text for the completed
  turn.
  """

  pass


# Tool
class PreToolCallDecideHook(DecideHook[types.ToolCall]):
  """Invoked before a tool call to decide if it should proceed.

  The `data` parameter receives the `types.ToolCall` object.
  Hooks can return:
  - `HookResult(allow=True)` to allow execution with original arguments.
  - `HookResult(allow=True, modified_args={...})` to allow execution with
    modified arguments.
  - `HookResult(allow=False, message="...")` to deny execution.
  """

  pass


class PostToolCallHook(InspectHook[types.ToolResult]):
  """Invoked after a tool call completes.

  The `data` parameter receives the `types.ToolResult` object containing the
  tool execution details.
  """

  pass


class OnToolErrorHook(TransformHook[Exception, Any]):
  """Invoked when a tool fails, allowing authors to shape the failure message.

  Receives the raised exception and returns an optional custom error string.
  If a non-empty string is returned, it replaces the default stacktrace or
  error message delivered to the model. If None is returned, default error
  formatting is used. Works symmetrically across built-in and custom tools.

  The hook cannot fix or retry the tool call on its own, but it can guide
  the agent toward a specific resolution.

  To customize failure messages or provide recovery fallbacks for your tool,
  you can either return a custom error string from this hook or do so directly
  within your tool implementation.
  """

  pass


# Interaction
class OnInteractionHook(
    TransformHook[AskQuestionInteractionSpec, QuestionHookResult]
):
  """Hook invoked when the agent needs user interaction.

  This is a superset of QuestionHook and handles all user interactions.
  """

  pass


# Compaction
class OnCompactionHook(InspectHook[types.Step]):
  """Invoked when a context compaction event occurs.

  Compaction is triggered by the harness when the context window exceeds the
  configured token threshold. This hook provides an observability point for
  logging, metrics, or UI notifications.
  """

  pass


# --- Decorator Factory ---


def _is_context_parameter(param: inspect.Parameter) -> bool:
  """Checks if a parameter represents the HookContext."""
  if param.name == "context":
    return True
  annotation = param.annotation
  if annotation is inspect.Parameter.empty:
    return False
  # Note: Exact HookContext matching is used rather than subclass inheritance
  # to avoid maintaining a list of derived types, as current scoping makes exact
  # matching sufficient.
  if annotation == HookContext:
    return True
  if isinstance(annotation, str):
    simple_name = annotation.split(".")[-1]
    if simple_name == "HookContext":
      return True
  return False


def _make_hook_decorator(hook_cls: type[Any], *, pass_data: bool = True):
  """Creates a decorator that wraps a function as a Hook subclass.

  Each decorator-created hook delegates its ``run()`` to the wrapped
  function and remains directly callable for convenience. Both synchronous
  and asynchronous functions are supported.

  If the wrapped function accepts `context` (either by name 'context' or
  by type hint matching HookContext), the HookContext will be passed to it.

  Args:
    hook_cls: The concrete Hook class to subclass.
    pass_data: If True, the wrapped function receives the hook's ``data``
      argument.  If False (e.g. session start/end), it is called with no
      arguments by default.

  Returns:
    A decorator that converts a function into a Hook instance.
  """

  class _FunctionHookWithData(hook_cls):
    """Internal hook implementation wrapping a decorated function."""

    def __init__(self, f, call_fn):
      self.f = f
      self._call_fn = call_fn
      functools.update_wrapper(self, f)

    async def run(self, context: HookContext, data: Any) -> Any:
      res = self._call_fn(context, data)
      if inspect.isawaitable(res):
        return await res
      return res

    async def __call__(self, *args, **kwargs):
      res = self.f(*args, **kwargs)
      if inspect.isawaitable(res):
        return await res
      return res

  class _FunctionHookNoData(hook_cls):
    """Internal hook implementation wrapping a decorated function."""

    def __init__(self, f, call_fn):
      self.f = f
      self._call_fn = call_fn
      functools.update_wrapper(self, f)

    async def run(self, context: HookContext, data: Any) -> Any:
      res = self._call_fn(context)
      if inspect.isawaitable(res):
        return await res
      return res

    async def __call__(self, *args, **kwargs):
      res = self.f(*args, **kwargs)
      if inspect.isawaitable(res):
        return await res
      return res

  def decorator(func):
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    num_params = len(params)

    # Find if there is a context parameter
    context_idx = -1
    for i, p in enumerate(params):
      if _is_context_parameter(p):
        context_idx = i
        break

    is_stateful = context_idx != -1

    # Early validation of function signature
    try:
      if pass_data:
        if is_stateful:
          if num_params >= 2:
            if context_idx == 0:
              sig.bind(object(), object())
            else:
              sig.bind(object(), **{params[context_idx].name: object()})
          else:
            sig.bind(object())
        else:
          sig.bind(object())
      else:
        if is_stateful:
          sig.bind(object())
        else:
          sig.bind()
    except TypeError as e:
      raise TypeError(
          f"Invalid signature for hook '{func.__name__}': {e}"
      ) from e

    if pass_data:
      if is_stateful:
        if num_params >= 2:
          if context_idx == 0:
            call_fn_data = func
          else:
            # Pass context as keyword argument to avoid positional conflicts
            context_name = params[context_idx].name
            call_fn_data = lambda ctx, d: func(d, **{context_name: ctx})
        else:
          # Only 1 param, and it is context (ignoring data)
          call_fn_data = lambda ctx, d: func(ctx)
      else:
        # Stateless, expects data
        call_fn_data = lambda ctx, d: func(d)

      return _FunctionHookWithData(func, call_fn_data)

    else:
      if is_stateful:
        call_fn_nodata = func
      else:
        call_fn_nodata = lambda ctx: func()

      return _FunctionHookNoData(func, call_fn_nodata)

  return decorator


# --- Decorators ---

pre_turn = _make_hook_decorator(PreTurnHook)
pre_tool_call_decide = _make_hook_decorator(PreToolCallDecideHook)
on_interaction = _make_hook_decorator(OnInteractionHook)
on_compaction = _make_hook_decorator(OnCompactionHook)
on_session_start = _make_hook_decorator(OnSessionStartHook, pass_data=False)
on_session_end = _make_hook_decorator(OnSessionEndHook, pass_data=False)
post_turn = _make_hook_decorator(PostTurnHook)
post_tool_call = _make_hook_decorator(PostToolCallHook)
on_tool_error = _make_hook_decorator(OnToolErrorHook)


# Internal hooks for telemetry
class _PreStepHook(InspectHook[types.Step]):
  """Invoked when a step is first seen in the stream (internal)."""


class _PostStepHook(InspectHook[types.Step]):
  """Invoked when a step completes (internal)."""


_pre_step = _make_hook_decorator(_PreStepHook)
_post_step = _make_hook_decorator(_PostStepHook)
