# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
from typing import Any
from uuid import uuid4
import asyncio

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})

    async def run(self, sampling_params: dict[str, Any], index:int, stream: bool = False, **kwargs) -> AgentLoopOutput:
        try:
            prompt_ids = kwargs.get("prompt_ids", [])
            response_ids = kwargs.get("response_ids", [])

            if len(prompt_ids) == 0:
                messages = list(kwargs["raw_prompt"])
                prompt_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
                    ),
                )
            input_ids = prompt_ids + response_ids
            metrics = {}
            request_id = uuid4().hex
            prompt_length = len(prompt_ids)
            with simple_timer("generate_sequences", metrics):
                task = asyncio.create_task(
                    self.server_manager.generate(
                    request_id=request_id, prompt_ids=input_ids, prompt_length=prompt_length, sampling_params=sampling_params, stream=stream
                ))
                delta_ids = await task

        except asyncio.CancelledError:
            try: task
            except NameError:
                prompt_ids = kwargs.get("prompt_ids", [])
                response_ids = kwargs.get("response_ids", [])
                delta_ids = []
                metrics = {}
            else:
                task.cancel()
                delta_ids = await task
        finally:
            response_ids = response_ids + delta_ids
            response_mask = [1] * len(response_ids)
            output = AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids[: self.response_length],
                response_mask=response_mask[: self.response_length],
                multi_modal_data={},
                num_turns=2,
                index=index,
                metrics=metrics,
            )
            return output
