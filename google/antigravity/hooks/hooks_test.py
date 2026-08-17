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

"""Tests for specific Hook interfaces and result types in v2."""

from typing import Any
import unittest

from google.antigravity import types
from google.antigravity.hooks import hooks


class BaseHookTest(unittest.IsolatedAsyncioTestCase):
  """Tests default behavior of specific Hook classes and result types."""

  def test_hook_result_defaults(self):
    """Verifies the default attributes of HookResult."""
    res = hooks.HookResult()
    self.assertTrue(res.allow)
    self.assertEqual(res.message, "")

  async def test_inspect_hook(self):
    """Verifies InspectHook can be executed."""

    class DummyInspectHook(hooks.InspectHook):

      async def run(self, context: hooks.HookContext, data: Any) -> None:
        data["called"] = True

    hook = DummyInspectHook()
    ctx = hooks.HookContext()
    data = {}
    await hook.run(ctx, data)
    self.assertTrue(data["called"])

  async def test_decide_hook(self):
    """Verifies DecideHook can be executed and returns HookResult."""

    class DummyDecideHook(hooks.DecideHook):

      async def run(
          self, context: hooks.HookContext, data: Any
      ) -> hooks.HookResult:
        return hooks.HookResult(allow=True, message="allowed")

    hook = DummyDecideHook()
    ctx = hooks.HookContext()
    res = await hook.run(ctx, None)
    self.assertTrue(res.allow)
    self.assertEqual(res.message, "allowed")

  async def test_pre_turn_hook(self):
    """Verifies PreTurnHook accepts types.Content and returns HookResult."""

    class DummyPreTurnHook(hooks.PreTurnHook):

      async def run(
          self, context: hooks.HookContext, data: Any
      ) -> hooks.HookResult:
        return hooks.HookResult(allow=isinstance(data, list), message="checked")

    hook = DummyPreTurnHook()
    ctx = hooks.HookContext()
    res = await hook.run(ctx, ["multimodal", "prompt"])
    self.assertTrue(res.allow)
    self.assertEqual(res.message, "checked")

  async def test_transform_hook(self):
    """Verifies TransformHook can be executed and modifies data."""

    class DummyTransformHook(hooks.TransformHook):

      async def run(self, context: hooks.HookContext, data: Any) -> Any:
        return data + "_modified"

    hook = DummyTransformHook()
    ctx = hooks.HookContext()
    res = await hook.run(ctx, "original")
    self.assertEqual(res, "original_modified")

  async def test_on_compaction_hook(self):
    """Verifies OnCompactionHook can be instantiated and executed."""
    called = False

    class DummyCompactionHook(hooks.OnCompactionHook):

      async def run(self, context: hooks.HookContext, data: types.Step) -> None:
        nonlocal called
        called = True

    hook = DummyCompactionHook()
    ctx = hooks.HookContext()
    step = types.Step(type=types.StepType.COMPACTION)
    await hook.run(ctx, step)
    self.assertTrue(called)

  async def test_decorator_pre_turn(self):
    """Verifies @pre_turn decorator works."""
    called_with = None

    @hooks.pre_turn
    async def my_hook(data):
      nonlocal called_with
      called_with = data
      return hooks.HookResult(allow=True)

    ctx = hooks.HookContext()
    res = await my_hook.run(ctx, "prompt_data")
    self.assertTrue(res.allow)
    self.assertEqual(called_with, "prompt_data")

  async def test_decorator_on_session_start(self):
    """Verifies @on_session_start decorator works (pass_data=False)."""
    called = False

    @hooks.on_session_start
    async def my_hook():
      nonlocal called
      called = True

    ctx = hooks.HookContext()
    await my_hook.run(ctx, None)
    self.assertTrue(called)

  async def test_decorator_pre_turn_with_context(self):
    """Verifies @pre_turn decorator works when accepting context."""
    called_with = None
    context_passed = None

    @hooks.pre_turn
    async def my_hook(context, data):
      nonlocal called_with, context_passed
      called_with = data
      context_passed = context
      return hooks.HookResult(allow=True)

    ctx = hooks.HookContext()
    res = await my_hook.run(ctx, "prompt_data")
    self.assertTrue(res.allow)
    self.assertEqual(called_with, "prompt_data")
    self.assertIs(context_passed, ctx)

  async def test_decorator_on_session_start_with_context(self):
    """Verifies @on_session_start decorator works when accepting context."""
    context_passed = None

    @hooks.on_session_start
    async def my_hook(context):
      nonlocal context_passed
      context_passed = context

    ctx = hooks.HookContext()
    await my_hook.run(ctx, None)
    self.assertIs(context_passed, ctx)

  async def test_decorator_pre_turn_compatibility_edge_case(self):
    """Verifies that 2-argument hooks without 'context' as first arg still work."""
    called_with = None
    extra_val = None

    @hooks.pre_turn
    async def my_hook(data, extra="default"):
      nonlocal called_with, extra_val
      called_with = data
      extra_val = extra
      return hooks.HookResult(allow=True)

    ctx = hooks.HookContext()
    res = await my_hook.run(ctx, "prompt_data")
    self.assertTrue(res.allow)
    self.assertEqual(called_with, "prompt_data")
    self.assertEqual(extra_val, "default")

  async def test_decorator_pre_turn_with_type_hint(self):
    """Verifies @pre_turn decorator works when accepting context named 'ctx' with type hint."""
    called_with = None
    context_passed = None

    @hooks.pre_turn
    async def my_hook(ctx: hooks.HookContext, data):
      nonlocal called_with, context_passed
      called_with = data
      context_passed = ctx
      return hooks.HookResult(allow=True)

    ctx = hooks.HookContext()
    res = await my_hook.run(ctx, "prompt_data")
    self.assertTrue(res.allow)
    self.assertEqual(called_with, "prompt_data")
    self.assertIs(context_passed, ctx)

  async def test_decorator_on_session_start_with_type_hint(self):
    """Verifies @on_session_start works when accepting context named 'ctx' with type hint."""
    context_passed = None

    @hooks.on_session_start
    async def my_hook(ctx: hooks.HookContext):
      nonlocal context_passed
      context_passed = ctx

    ctx = hooks.HookContext()
    await my_hook.run(ctx, None)
    self.assertIs(context_passed, ctx)

  async def test_decorator_pre_turn_reversed_order(self):
    """Verifies @pre_turn works when context is placed second with type hint."""
    called_with = None
    context_passed = None

    @hooks.pre_turn
    async def my_hook(data, ctx: hooks.HookContext):
      nonlocal called_with, context_passed
      called_with = data
      context_passed = ctx
      return hooks.HookResult(allow=True)

    ctx = hooks.HookContext()
    res = await my_hook.run(ctx, "prompt_data")
    self.assertTrue(res.allow)
    self.assertEqual(called_with, "prompt_data")
    self.assertIs(context_passed, ctx)

  async def test_decorator_pre_turn_with_string_type_hint(self):
    """Verifies @pre_turn works when context has a string type hint."""
    called_with = None
    context_passed = None

    @hooks.pre_turn
    async def my_hook(ctx: "HookContext", data):
      nonlocal called_with, context_passed
      called_with = data
      context_passed = ctx
      return hooks.HookResult(allow=True)

    ctx = hooks.HookContext()
    res = await my_hook.run(ctx, "prompt_data")
    self.assertTrue(res.allow)
    self.assertEqual(called_with, "prompt_data")
    self.assertIs(context_passed, ctx)

  def test_decorator_invalid_signature_fails_early(self):
    """Verifies that invalid decorator signatures raise TypeError early."""
    # pylint: disable=unused-argument

    # pre_turn expects data (pass_data=True). If we define with 0 params, it
    # should fail.
    with self.assertRaises(TypeError):
      @hooks.pre_turn
      async def _no_args():
        pass

    # pre_turn with too many required arguments (without defaults)
    with self.assertRaises(TypeError):
      @hooks.pre_turn
      async def _too_many(ctx: hooks.HookContext, data, extra_required):
        pass

    # on_session_start does not expect data (pass_data=False).
    # If we define it with 2 params (no defaults), it should fail.
    with self.assertRaises(TypeError):
      @hooks.on_session_start
      async def _bad_session(ctx: hooks.HookContext, extra_required):
        pass

  async def test_decorator_pre_turn_optional_param_before_context(self):
    """Verifies @pre_turn works with optional parameter before context."""
    called_with = None
    context_passed = None
    extra_val = None

    @hooks.pre_turn
    async def my_hook(data, extra="default", ctx: hooks.HookContext = None):
      nonlocal called_with, context_passed, extra_val
      called_with = data
      context_passed = ctx
      extra_val = extra
      return hooks.HookResult(allow=True)

    ctx = hooks.HookContext()
    res = await my_hook.run(ctx, "prompt_data")
    self.assertTrue(res.allow)
    self.assertEqual(called_with, "prompt_data")
    self.assertEqual(extra_val, "default")
    self.assertIs(context_passed, ctx)

  async def test_decorator_pre_turn_context_only(self):
    """Verifies @pre_turn works when only accepting context (ignoring data)."""
    context_passed = None

    @hooks.pre_turn
    async def my_hook(ctx: hooks.HookContext):
      nonlocal context_passed
      context_passed = ctx
      return hooks.HookResult(allow=True)

    ctx = hooks.HookContext()
    res = await my_hook.run(ctx, "prompt_data")
    self.assertTrue(res.allow)
    self.assertIs(context_passed, ctx)

  async def test_decorator_sync_pre_turn(self):
    """Verifies synchronous @pre_turn hook runs correctly."""
    called_with = None

    @hooks.pre_turn
    def my_sync_hook(data):
      nonlocal called_with
      called_with = data
      return hooks.HookResult(allow=True, message="sync_allowed")

    ctx = hooks.HookContext()
    res = await my_sync_hook.run(ctx, "prompt_data")
    self.assertTrue(res.allow)
    self.assertEqual(res.message, "sync_allowed")
    self.assertEqual(called_with, "prompt_data")

  async def test_decorator_sync_on_session_start(self):
    """Verifies synchronous @on_session_start hook runs correctly."""
    called = False

    @hooks.on_session_start
    def my_sync_hook():
      nonlocal called
      called = True

    ctx = hooks.HookContext()
    await my_sync_hook.run(ctx, None)
    self.assertTrue(called)

  async def test_decorator_sync_pre_turn_with_context(self):
    """Verifies synchronous @pre_turn hook with context parameter works."""
    called_with = None
    context_passed = None

    @hooks.pre_turn
    def my_sync_hook(ctx: hooks.HookContext, data):
      nonlocal called_with, context_passed
      called_with = data
      context_passed = ctx
      return hooks.HookResult(allow=True)

    ctx = hooks.HookContext()
    res = await my_sync_hook.run(ctx, "prompt_data")
    self.assertTrue(res.allow)
    self.assertEqual(called_with, "prompt_data")
    self.assertIs(context_passed, ctx)

  async def test_decorator_sync_direct_call(self):
    """Verifies calling a synchronous decorated hook directly works."""

    @hooks.pre_turn
    def my_sync_hook(data):
      return f"echo: {data}"

    res = await my_sync_hook("test")
    self.assertEqual(res, "echo: test")


if __name__ == "__main__":
  unittest.main()
