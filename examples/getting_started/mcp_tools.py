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

"""MCP Integration example for Google Antigravity SDK.

This example demonstrates how to connect an agent to external MCP servers
using stdio, SSE, and Streamable HTTP transports.

To run:
  python mcp_tools.py

Criteria for correct script performance:
  1. The script exits cleanly with return code 0 (no unhandled exceptions).
  2. The agent successfully uses the pirate_multiply tool to multiply numbers.
  3. The agent respects the safety policy and fails to use the pirate_divide
     tool when it is denied.
"""

import asyncio
import os

from google.antigravity import Agent
from google.antigravity import LocalAgentConfig
from google.antigravity import types

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from resources import mcp_server
from google.antigravity.hooks import policy


async def mcp_stdio() -> None:
  """Showcases the Stdio transport."""
  print("\n  --- Showcasing Stdio Transport ---")
  mcp_server_path = os.path.join(
      os.path.dirname(__file__), "..", "resources", "mcp_server.py"
  )
  stdio_server = types.McpStdioServer(
      name="pirate_math",
      command="python3",
      args=[mcp_server_path, "--transport=stdio"],
  )

  config = LocalAgentConfig(mcp_servers=[stdio_server])

  async with Agent(config) as my_agent:
    prompt = "Use the pirate_multiply tool to multiply 5 and 7."
    print(f"  User: {prompt}")
    response = await my_agent.chat(prompt)
    print(f"  Agent: {await response.text()}")


async def mcp_http() -> None:
  """Showcases the Streamable HTTP transport."""
  print("\n  --- Showcasing Streamable HTTP Transport ---")
  async with mcp_server.run("streamable-http") as port:
    config = LocalAgentConfig(
        mcp_servers=[
            types.McpStreamableHttpServer(
                name="pirate_math",
                url=f"http://localhost:{port}/mcp",
            )
        ]
    )

    async with Agent(config) as my_agent:
      prompt = "Use the pirate_multiply tool to multiply 5 and 7."
      print(f"  User: {prompt}")
      response = await my_agent.chat(prompt)
      print(f"  Agent: {await response.text()}")


async def mcp_filtering() -> None:
  """Showcases MCP tool filtering (enabled_tools / disabled_tools)."""
  print("\n  --- Showcasing MCP Tool Filtering (disabled_tools) ---")
  mcp_server_path = os.path.join(
      os.path.dirname(__file__), "..", "resources", "mcp_server.py"
  )
  stdio_server = types.McpStdioServer(
      name="pirate_math",
      command="python3",
      args=[mcp_server_path, "--transport=stdio"],
      disabled_tools=["pirate_divide"],
  )

  config = LocalAgentConfig(mcp_servers=[stdio_server])

  async with Agent(config) as my_agent:
    # The pirate_multiply tool should work
    prompt1 = "Use the pirate_multiply tool to multiply 6 and 8."
    print(f"  User: {prompt1}")
    response1 = await my_agent.chat(prompt1)
    print(f"  Agent: {await response1.text()}")

    # The pirate_divide tool is disabled/removed from the model's context,
    # so the model should fail or state it cannot divide.
    prompt2 = "Use the pirate_divide tool to divide 10 by 2."
    print(f"\n  User: {prompt2}")
    response2 = await my_agent.chat(prompt2)
    print(f"  Agent: {await response2.text()}")


async def mcp_policies() -> None:
  """Showcases safety policies for MCP tools using new overloads."""
  print("\n  --- Showcasing MCP Safety Policies ---")
  mcp_server_path = os.path.join(
      os.path.dirname(__file__), "..", "resources", "mcp_server.py"
  )
  stdio_server = types.McpStdioServer(
      name="pirate_math",
      command="python3",
      args=[mcp_server_path, "--transport=stdio"],
  )

  # Define safety policies. Note that we can now pass `stdio_server` (BaseMcpServerConfig)
  # directly to the policy builders!
  policies = [
      policy.deny_all(),
      policy.allow(stdio_server, ["pirate_multiply"]),  # Allow pirate_multiply
      policy.deny(stdio_server, ["pirate_divide"]),  # Deny pirate_divide
  ]

  config = LocalAgentConfig(mcp_servers=[stdio_server], policies=policies)

  async with Agent(config) as my_agent:
    # Multiply is allowed
    prompt1 = "Multiply 4 and 9 using the pirate_multiply tool."
    print(f"  User: {prompt1}")
    response1 = await my_agent.chat(prompt1)
    print(f"  Agent: {await response1.text()}")

    # Divide is denied by policy (visible to agent, but blocked at runtime)
    prompt2 = "Divide 12 by 3 using the pirate_divide tool."
    print(f"\n  User: {prompt2}")
    response2 = await my_agent.chat(prompt2)
    print(f"  Agent: {await response2.text()}")


async def main() -> None:
  await mcp_stdio()
  await mcp_filtering()
  await mcp_policies()
  await mcp_http()


if __name__ == "__main__":
  asyncio.run(main())
