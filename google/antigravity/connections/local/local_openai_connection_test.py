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

"""Unit tests for LocalOpenAIConnectionStrategy and LocalOpenAIAgentConfig."""

import unittest
from unittest import mock

from google.antigravity.proto import localharness_pb2
from google.antigravity import types
from google.antigravity.connections.local import local_openai_connection
from google.antigravity.connections.local import local_openai_connection_config
from google.antigravity.connections.local import test_utils


class LocalOpenAIConnectionTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    test_utils.patch_default_binary_path(self)

  def test_local_openai_strategy_harness_config(self):
    """Verify generic external OpenAI configuration works and clears Gemini config."""
    config = local_openai_connection_config.LocalOpenAIAgentConfig(
        base_url="http://localhost:11434/v1",
        model="llama3.1",
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )

    self.assertIsInstance(
        strategy, local_openai_connection.LocalOpenAIConnectionStrategy
    )
    h_cfg = strategy._build_harness_config()

    self.assertEqual(len(h_cfg.models), 1)
    model = h_cfg.models[0]
    self.assertEqual(model.name, "llama3.1")
    self.assertEqual(model.types, [localharness_pb2.MODEL_TYPE_TEXT])
    self.assertTrue(model.HasField("gemma_endpoint"))
    self.assertEqual(model.gemma_endpoint.base_url, "http://localhost:11434/v1")

  def test_local_openai_strategy_validate_empty_base_url(self):
    """Verify LocalOpenAIConnectionStrategy validates non-empty base_url."""
    strategy = local_openai_connection.LocalOpenAIConnectionStrategy(
        base_url="",
        model_name="test",
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    with self.assertRaises(types.AntigravityValidationError):
      strategy._validate_connection()

  def test_local_openai_config_model_target_parsing(self):
    """Verify LocalOpenAIAgentConfig parses model and endpoint base_url from ModelTarget."""
    endpoint = types.GeminiAPIEndpoint(base_url="http://custom-ollama:11434/v1")
    target = types.ModelTarget(name="llama3.2", endpoint=endpoint)
    config = local_openai_connection_config.LocalOpenAIAgentConfig(model=target)
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    self.assertEqual(strategy._model_name, "llama3.2")
    self.assertEqual(strategy._base_url, "http://custom-ollama:11434/v1")

  def test_local_openai_config_default_capabilities(self):
    """Verify LocalOpenAIAgentConfig defaults to all capabilities enabled."""
    openai_config = local_openai_connection_config.LocalOpenAIAgentConfig(
        base_url="http://localhost:11434/v1",
        model="llama3.1",
    )
    self.assertIsNone(openai_config.capabilities.enabled_tools)
    self.assertIsNone(openai_config.capabilities.disabled_tools)

  def test_local_openai_config_workspace_policies(self):
    """Verify LocalOpenAIAgentConfig does not prepend workspace_only policy."""
    config_openai = local_openai_connection_config.LocalOpenAIAgentConfig(
        base_url="http://localhost",
        model="m",
        workspaces=["/tmp/my_workspace"],
    )
    # LocalOpenAI uses localharness for native workspace containment;
    # config_openai.policies contains only the 2 confirm_run_command policies.
    self.assertEqual(len(config_openai.policies), 2)
    self.assertEqual(config_openai.policies[0].tool, "run_command")
    self.assertEqual(config_openai.policies[1].tool, "*")

  def test_local_openai_config_mcp_servers_and_subagents_passed_to_strategy(
      self,
  ):
    """Verify LocalOpenAIAgentConfig passes mcp_servers and subagents to strategy."""
    mcp_server = types.McpStdioServer(
        name="test_mcp", command="echo", args=["hello"]
    )
    subagent = types.SubagentConfig(
        name="test_subagent",
        description="A test subagent",
        system_instructions="You are a subagent",
    )
    config = local_openai_connection_config.LocalOpenAIAgentConfig(
        base_url="http://localhost:11434/v1",
        model="llama3.1",
        mcp_servers=[mcp_server],
        subagents=[subagent],
    )
    strategy = config.create_strategy(
        tool_runner=mock.MagicMock(),
        hook_runner=mock.MagicMock(),
    )
    self.assertEqual(strategy._mcp_servers, [mcp_server])
    self.assertEqual(strategy._subagents, [subagent])


if __name__ == "__main__":
  unittest.main()
