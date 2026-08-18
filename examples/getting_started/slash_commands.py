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

"""Example demonstrating Plan Slash Command (/plan) in the SDK.

This example showcases:
- Sending the `/plan` slash command programmatically using the
  `types.SlashCommand` object in a prompt parts list.
- Understanding how planning mode generates a detailed implementation plan and
  awaits user approval (human-in-the-loop).

Criteria for correct script performance:
  1. The script exits cleanly with return code 0 (no unhandled exceptions).
  2. The agent successfully generates an implementation plan artifact.
  3. The script prints a success message confirming the plan exists.
"""

import asyncio
import os
import tempfile

from google.antigravity import Agent
from google.antigravity import LocalAgentConfig
from google.antigravity import types
from google.antigravity.hooks import policy


async def print_response_chunks(response) -> None:
  """Streams and prints response chunks separating thoughts and text visually."""
  current_type = None
  async for chunk in response.chunks:
    if isinstance(chunk, types.Thought):
      if current_type != "thought":
        if current_type is not None:
          print()
        print("\033[90m💭 [Thought]: ", end="", flush=True)
        current_type = "thought"
      print(chunk.text, end="", flush=True)
    elif isinstance(chunk, types.Text):
      if current_type != "text":
        if current_type == "thought":
          print("\033[0m")  # Reset ANSI formatting
        if current_type is not None:
          print()
        print("\033[32m💬 [Response]:\033[0m ", end="", flush=True)
        current_type = "text"
      print(chunk.text, end="", flush=True)
  if current_type == "thought":
    print("\033[0m")
  print()


async def main() -> None:
  # Use a temporary directory for workspace and app_data_dir to prevent clutter
  # and ensure write access in all testing environments.
  with tempfile.TemporaryDirectory() as tmpdir:
    # Enable safety policy to allow all tools, including file creation and
    # execution, which planning agent uses to create plans/scripts.
    policies = [policy.allow_all()]

    # workspaces: The directory where the agent reads and writes files.
    # app_data_dir: Where the agent stores session data (e.g. brain artifacts).
    # policies: Allows autonomous tool execution without user prompts.
    config = LocalAgentConfig(
        workspaces=[tmpdir],
        app_data_dir=tmpdir,
        policies=policies,
    )

    # Programmatic slash command usage with /plan:
    # You can explicitly pass a `types.SlashCommand` primitive in the list
    # of prompt parts.
    async with Agent(config) as my_agent:
      print("--- Programmatic Plan Slash Command (Part Object) ---")
      prompt = [
          types.SlashCommand(name=types.BuiltinSlashCommandName.PLAN),
          "Write a python script that prints numbers 1 to 10.",
      ]
      print(f"Sending prompt parts: {prompt}\n")

      response = await my_agent.conversation.chat(prompt)
      await print_response_chunks(response)

      print("Asking the agent for the path to the generated plan...")
      path_prompt = (
          "What is the absolute path of the implementation plan artifact you"
          " just created? Please return ONLY the absolute file path as a plain"
          " string, with no markdown formatting, no backticks, and no"
          " additional text."
      )
      path_response = await my_agent.conversation.chat(path_prompt)

      plan_path = await path_response.text()
      plan_path = plan_path.strip().strip("`").strip('"').strip("'")
      print(f"Agent reported plan path: {plan_path}\n")

      if os.path.exists(plan_path):
        print(f"✅ Success: Verified plan exists at path: {plan_path}\n")
        print("--- First 5 lines of generated plan ---")
        with open(plan_path, "r", encoding="utf-8") as f:
          for _ in range(5):
            line = f.readline()
            if not line:
              break
            print(f"  {line.rstrip()}")
        print("----------------------------------------\n")
      else:
        print(
            "❌ Error: File does not exist at agent-reported path:"
            f" '{plan_path}'!\n"
        )
        exit(1)


if __name__ == "__main__":
  asyncio.run(main())
