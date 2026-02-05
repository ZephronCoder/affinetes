"""
QQR-based Travel Planning Evaluation Environment - Actor Implementation.

Fully reuses QQR's MCP tool system for real tool invocation.

Core reuse:
- qqr.rollout.agent_rollout.MCPState - MCP server state management (singleton)
- qqr.mcp.MCPServerStdioCacheable - Cacheable MCP server
- qqr.tools.amap - AMap tools server
- qqr.tools.mock_transport - Mock transport tools server

Scoring structure:
- Hard constraints: format, tool info usage, required tools called, tool quality gate
- Code score (70 pts): info consistency(35), completeness(35) with grounding verification
- LLM score (30 pts): smooth coupling with code score via LLM_CODE_RATIO_FACTOR
"""

import json
import os
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List

import httpx

# ==================== Reuse QQR's MCP System ====================
# Import from mcp_wrapper (copied from QQR to avoid slime dependency)
from mcp_wrapper import MCPServerStdioCacheable, MCPServerStdioParams, MCPState

# Import from local modules
from config import (
    SYSTEM_PROMPT,
    MAX_TOOL_STEPS,
    TOOLS_SCHEMA,
    CHUTES_API_KEY,
    AMAP_MAPS_API_KEY,
    PYTHONPATH,
    REQUIRED_TOOLS_BY_TYPE,
    REQUIRES_TRANSPORT,
)
from problem_generator import TravelProblem, get_generator
from scorer import TravelScorer
from llm_validator import LLMValidator

from affinetes.core.openenv import OpenEnvResponse

# ==================== MCP Server Configuration (QQR Format) ====================
def mcp_server_config_fn() -> list:
    """
    MCP server configuration function.

    Returns QQR-format MCP server list with AMap and Transport only.
    API keys are passed via environment variables.
    """
    # AMap server configuration
    amap_server_params = MCPServerStdioParams(
        command="python",
        args=["-m", "qqr.tools.amap"],
        env={
            "AMAP_MAPS_API_KEY": AMAP_MAPS_API_KEY or "",
            "PYTHONPATH": PYTHONPATH or "",
        },
    )
    amap_server = MCPServerStdioCacheable(
        name="AMap",
        params=amap_server_params,
        cache_tools_list=True,
        client_session_timeout_seconds=60,
        max_retry_attempts=3,
        blocklist=[],
        cache_ttl=172800,
        cache_maxsize=32768,
        concurrency_limit=16,
    )

    # Transport server configuration
    # Uses deterministic algorithmic generation (no LLM dependency)
    # TRANSPORT_SALT changes weekly for anti-memorization
    epoch_salt = os.getenv("TRANSPORT_SALT", str(int(time.time()) // (7 * 86400)))
    transport_server_params = MCPServerStdioParams(
        command="python",
        args=["-m", "mock_transport.server"],
        env={
            "PYTHONPATH": PYTHONPATH or "",
            "TRANSPORT_SALT": epoch_salt,
        },
    )
    transport_server = MCPServerStdioCacheable(
        name="Transport",
        params=transport_server_params,
        cache_tools_list=True,
        client_session_timeout_seconds=120,
        max_retry_attempts=3,
        blocklist=[],
        cache_ttl=172800,  # 48 hours
        cache_maxsize=32768,  # 4x expanded
        concurrency_limit=4,
    )

    return [amap_server, transport_server]


# ==================== Step Reward Calculator ====================
class StepRewardCalculator:
    """Calculate intermediate rewards after each tool call step."""

    # Tools that are appropriate at different stages
    EARLY_TOOLS = {"poi_search", "weather", "search_flights", "search_train_tickets"}
    MID_TOOLS = {"around_search", "direction"}

    @staticmethod
    def calculate_step_reward(
        tool_name: str,
        tool_args: Dict,
        tool_result: str,
        problem: TravelProblem,
        step_number: int,
        called_tools_so_far: set,
    ) -> float:
        """
        Calculate reward for a single tool call step.

        Returns a float in [0, 1]:
          0.4 * tool_selection_score
          + 0.3 * arg_quality_score
          + 0.3 * result_usefulness_score
        """
        tool_sel = StepRewardCalculator._tool_selection_score(
            tool_name, problem, step_number, called_tools_so_far
        )
        arg_qual = StepRewardCalculator._arg_quality_score(tool_name, tool_args)
        result_use = StepRewardCalculator._result_usefulness_score(tool_result)

        return round(0.4 * tool_sel + 0.3 * arg_qual + 0.3 * result_use, 4)

    @staticmethod
    def _tool_selection_score(
        tool_name: str,
        problem: TravelProblem,
        step_number: int,
        called_tools_so_far: set,
    ) -> float:
        """Is this the right tool to call at this step?"""
        required = set(problem.required_tools)
        score = 0.0

        # Calling a required tool earns base score
        if tool_name in required:
            score += 0.6
        else:
            score += 0.2  # Non-required but possibly useful

        # Bonus for not repeating the same tool
        if tool_name not in called_tools_so_far:
            score += 0.2

        # Bonus for calling transport tools when problem requires transport
        if problem.problem_type in REQUIRES_TRANSPORT:
            if tool_name in {"search_flights", "search_train_tickets"}:
                score += 0.2

        return min(1.0, score)

    @staticmethod
    def _arg_quality_score(tool_name: str, args: Dict) -> float:
        """Are the arguments well-formed?"""
        if not args:
            return 0.0

        if tool_name == "poi_search":
            address = args.get("address", "")
            if address and len(address) >= 2:
                return 1.0 if re.search(r'[\u4e00-\u9fa5]', address) else 0.5
            return 0.2

        elif tool_name == "direction":
            origin = args.get("origin", "")
            dest = args.get("destination", "")
            coord_re = re.compile(r'^\d+\.\d+\s*,\s*\d+\.\d+$')
            origin_ok = bool(coord_re.match(origin.strip())) if origin else False
            dest_ok = bool(coord_re.match(dest.strip())) if dest else False
            if origin_ok and dest_ok:
                return 1.0
            elif origin_ok or dest_ok:
                return 0.5
            return 0.2

        elif tool_name == "around_search":
            loc = args.get("location", "")
            coord_re = re.compile(r'^\d+\.\d+\s*,\s*\d+\.\d+$')
            return 1.0 if (loc and coord_re.match(loc.strip())) else 0.3

        elif tool_name == "weather":
            city = args.get("city", "")
            return 1.0 if (city and re.match(r'^[\u4e00-\u9fa5]{2,10}$', city)) else 0.3

        elif tool_name in ("search_flights", "search_train_tickets"):
            date = args.get("date", "")
            from_c = args.get("from_city", "")
            to_c = args.get("to_city", "")
            date_ok = bool(re.match(r'^\d{4}-\d{2}-\d{2}$', date))
            cities_ok = bool(from_c and to_c)
            if date_ok and cities_ok:
                return 1.0
            elif date_ok or cities_ok:
                return 0.5
            return 0.2

        return 0.5  # Unknown tool, neutral

    @staticmethod
    def _result_usefulness_score(result: str) -> float:
        """Is the tool result useful (non-empty, non-error)?"""
        if not result or len(result.strip()) < 10:
            return 0.0

        # Check for error patterns
        result_lower = result.lower()
        if len(result) < 100:
            error_keywords = ["error", "failed", "失败", "错误", "无效", "无法"]
            if any(kw in result_lower for kw in error_keywords):
                return 0.1

        # Longer results with content are more useful
        if len(result) > 500:
            return 1.0
        elif len(result) > 100:
            return 0.8
        return 0.5


@dataclass
class EpisodeState:
    """Episode state container."""
    episode_id: str
    task_id: int
    seed: int
    problem: TravelProblem
    conversation: List[Dict[str, Any]] = field(default_factory=list)
    tool_trace: List[Dict] = field(default_factory=list)
    step_rewards: List[float] = field(default_factory=list)
    current_step: int = 0
    done: bool = False
    final_score: float = 0.0
    score_breakdown: Optional[Dict] = None


class Actor:
    """
    QQR-based travel planning evaluation environment.

    Fully reuses QQR's MCPState to manage and invoke MCP tools.
    """

    def __init__(
        self,
        enable_llm_validator: bool = True,
        llm_validator_model: str = "Qwen/Qwen2.5-72B-Instruct",
    ):
        self._episodes: Dict[str, EpisodeState] = {}
        self._generator = get_generator()

        # ==================== Reuse QQR's MCPState ====================
        # MCPState is singleton, initialized with our config function
        self._mcp_state = MCPState(mcp_server_config_fn)
        self._mcp_initialized = False

        # LLM Validator
        self._llm_validator = None
        if enable_llm_validator:
            api_key = os.getenv("CHUTES_API_KEY")
            if api_key:
                self._llm_validator = LLMValidator(
                    model=llm_validator_model,
                    base_url="https://llm.chutes.ai/v1",
                    api_key=api_key,
                )

        self._scorer = TravelScorer(llm_validator=self._llm_validator)

    async def _ensure_mcp_initialized(self):
        """Ensure MCP servers are initialized."""
        if not self._mcp_initialized:
            await self._mcp_state.get_mcp_servers()
            self._mcp_initialized = True

    async def reset(
        self,
        task_id: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> OpenEnvResponse:
        """Reset environment and generate new problem."""
        # Ensure MCP servers are initialized
        await self._ensure_mcp_initialized()

        seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        task_id = task_id if task_id is not None else (seed & 0x7FFFFFFF)

        problem = self._generator.generate(task_id)
        prompt = self._generator.to_prompt(problem)

        episode_id = uuid.uuid4().hex
        ep = EpisodeState(
            episode_id=episode_id,
            task_id=task_id,
            seed=seed,
            problem=problem,
            conversation=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        self._episodes[episode_id] = ep

        return OpenEnvResponse(
            observation=prompt,
            reward=0.0,
            done=False,
            episode_id=episode_id,
            info={
                "task_id": task_id,
                "seed": seed,
                "problem_type": problem.problem_type,
                "problem": problem.to_dict(),
                "available_tools": [t["function"]["name"] for t in self._mcp_state.tools],
            },
        )

    async def step(
        self,
        action: str,
        episode_id: Optional[str] = None,
        tool_calls: Optional[List[Dict]] = None,
    ) -> OpenEnvResponse:
        """
        Execute one step.

        Args:
            action: Model output text content
            episode_id: Episode ID
            tool_calls: OpenAI format tool call list
        """
        ep = self._episodes.get(episode_id)
        if ep is None or ep.done:
            return OpenEnvResponse(
                observation="Error: Invalid episode",
                reward=0.0,
                done=True,
                episode_id=episode_id
            )

        ep.current_step += 1

        # If tool calls exist, execute using MCPState
        print(f"[DEBUG] step: tool_calls={len(tool_calls) if tool_calls else 0}, current_step={ep.current_step}, MAX={MAX_TOOL_STEPS}")
        if tool_calls and ep.current_step <= MAX_TOOL_STEPS:
            # Chutes API workaround: Convert assistant's tool_calls to text format
            # because Chutes API doesn't support OpenAI's tool_calls format
            tool_call_descriptions = []
            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "unknown")
                args = func.get("arguments", "{}")
                tool_call_descriptions.append(f"调用工具: {name}({args})")

            assistant_content = action or ""
            if tool_call_descriptions:
                if assistant_content:
                    assistant_content += "\n\n"
                assistant_content += "正在调用以下工具:\n" + "\n".join(tool_call_descriptions)

            ep.conversation.append({"role": "assistant", "content": assistant_content})
            print(f"[DEBUG] Assistant message (tool calls converted to text)")
        else:
            # No tool calls, add assistant message directly
            ep.conversation.append({"role": "assistant", "content": action or ""})

        if tool_calls and ep.current_step <= MAX_TOOL_STEPS:
            tool_results = []

            for call in tool_calls:
                func = call.get("function", {})
                name = func.get("name", "")
                call_id = call.get("id", str(uuid.uuid4()))

                # Parse arguments
                args_str = func.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except json.JSONDecodeError:
                    args = {}

                # ==================== Use QQR's MCPState to call tools ====================
                # Construct OpenAI format tool_call
                tool_call_dict = {
                    "id": call_id,
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                }

                # Call MCPState.call_tool
                try:
                    tool_response = await self._mcp_state.call_tool(tool_call_dict)
                    result_content = tool_response.get("content", "")
                except Exception as e:
                    print(f"[ERROR] Tool call {name} failed: {e}")
                    result_content = f"工具调用失败: {str(e)[:200]}"

                # Record tool call
                ep.tool_trace.append({
                    "name": name,
                    "arguments": args,
                    "result": {"text": result_content},
                })

                # Calculate step reward
                called_tools_so_far = set(t["name"] for t in ep.tool_trace[:-1])
                step_reward = StepRewardCalculator.calculate_step_reward(
                    tool_name=name,
                    tool_args=args,
                    tool_result=result_content,
                    problem=ep.problem,
                    step_number=ep.current_step,
                    called_tools_so_far=called_tools_so_far,
                )
                ep.step_rewards.append(step_reward)

                tool_results.append({
                    "tool": name,
                    "call_id": call_id,
                    "result": result_content[:2000] if len(result_content) > 2000 else result_content
                })

            # Combine all tool results into one user message (Chutes API workaround)
            # because Chutes API may not fully support tool role
            combined_results = "\n\n".join(
                f"[工具调用: {r['tool']}]\n结果:\n{r['result']}"
                for r in tool_results
            )

            # Check which required tools haven't been called
            called_tool_names = set(t["name"] for t in ep.tool_trace)
            required_tools = set(ep.problem.required_tools)
            missing_tools = required_tools - called_tool_names

            if missing_tools:
                hint = f"\n\n**注意**：你还需要调用以下工具获取完整信息：{', '.join(missing_tools)}。请继续调用这些工具，然后再提供最终规划方案。"
            else:
                hint = "\n\n所有必需工具已调用完成，请根据以上信息提供详细的规划方案。"

            ep.conversation.append({
                "role": "user",
                "content": f"以下是工具调用结果：\n\n{combined_results}{hint}"
            })

            observation = self._format_tool_results(tool_results)
            # Average step reward for this batch of tool calls
            avg_step_reward = sum(ep.step_rewards[-len(tool_calls):]) / len(tool_calls) if tool_calls else 0.0
            return OpenEnvResponse(
                observation=observation,
                reward=avg_step_reward,
                done=False,
                episode_id=episode_id,
                info={
                    "step": ep.current_step,
                    "tool_calls": len(tool_calls),
                    "step_reward": avg_step_reward,
                    "step_rewards_so_far": ep.step_rewards[:],
                },
            )
        else:
            # No tool calls or max steps reached, perform scoring
            ep.done = True

            # If action is empty or too short (e.g. "让我查一下" from max steps),
            # fall back to last substantive assistant message from conversation.
            # For messages with tool calls, extract content before "正在调用以下工具".
            scoring_input = action or ""
            if not scoring_input.strip() or (
                len(scoring_input.strip()) < 100
                and not re.search(r'(第.天|Day\s*\d|航班|火车|景点)', scoring_input)
            ):
                for msg in reversed(ep.conversation):
                    if msg.get("role") == "assistant" and msg.get("content", "").strip():
                        content = msg["content"]
                        # Extract content before tool call descriptions
                        if "正在调用以下工具" in content:
                            content = content.split("正在调用以下工具")[0].strip()
                        if len(content) > len(scoring_input):
                            scoring_input = content
                            if len(content) >= 100:
                                break  # Found substantive content

            try:
                score_result = await self._scorer.score(
                    scoring_input, ep.problem, ep.tool_trace
                )
                ep.final_score = score_result.total
                ep.score_breakdown = score_result.to_dict()
                print(f"[DEBUG] Scoring completed: total={score_result.total}, breakdown={ep.score_breakdown}")
            except Exception as e:
                print(f"[ERROR] Scoring failed: {e}")
                import traceback
                traceback.print_exc()
                ep.final_score = 0.0
                ep.score_breakdown = {"error": str(e)}

            return OpenEnvResponse(
                observation="完成",
                reward=ep.final_score / 100.0,
                done=True,
                episode_id=episode_id,
                info={
                    "score": ep.final_score,
                    "score_breakdown": ep.score_breakdown
                },
            )

    async def evaluate(
        self,
        model: str = "moonshotai/Kimi-K2.5-TEE",
        base_url: str = "https://llm.chutes.ai/v1",
        task_id: Optional[int] = None,
        seed: Optional[int] = None,
        timeout: int = 300,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Complete evaluation flow using OpenAI Function Calling."""
        start_time = datetime.now()
        seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        task_id = task_id if task_id is not None else (seed & 0x7FFFFFFF)
        api_key = api_key or os.getenv("CHUTES_API_KEY")

        reset_resp = await self.reset(task_id=task_id, seed=seed)
        episode_id = reset_resp.episode_id
        ep = self._episodes[episode_id]

        total_usage = {"prompt_tokens": 0, "completion_tokens": 0}

        async with httpx.AsyncClient(timeout=float(timeout)) as client:
            for _ in range(MAX_TOOL_STEPS + 2):
                # Call LLM
                response = await self._call_llm(
                    client=client,
                    messages=ep.conversation,
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    temperature=temperature,
                    timeout=timeout,
                    tools=TOOLS_SCHEMA,
                )

                if response is None:
                    print(f"[DEBUG] LLM returned None, triggering scoring with existing conversation")
                    # Trigger scoring with whatever conversation exists
                    step_resp = await self.step(action="", episode_id=episode_id, tool_calls=None)
                    break

                content = response.get("content") or ""
                tool_calls = response.get("tool_calls")
                usage = response.get("usage", {})
                print(f"[DEBUG] LLM response: content_len={len(content)}, tool_calls={len(tool_calls) if tool_calls else 0}")
                if tool_calls:
                    print(f"[DEBUG] tool_calls sample: {tool_calls[0] if tool_calls else 'none'}")

                if usage:
                    total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                    total_usage["completion_tokens"] += usage.get("completion_tokens", 0)

                # Execute one step
                step_resp = await self.step(
                    action=content,
                    episode_id=episode_id,
                    tool_calls=tool_calls,
                )

                if step_resp.done:
                    break
            else:
                # Loop exhausted without scoring — model kept calling tools
                # Force scoring with whatever conversation exists
                if not step_resp or not step_resp.done:
                    step_resp = await self.step(action="", episode_id=episode_id, tool_calls=None)

        final_ep = self._episodes.get(episode_id)
        if not final_ep:
            return {
                "task_name": "qqr",
                "score": 0.0,
                "success": False,
                "time_taken": (datetime.now() - start_time).total_seconds(),
                "extra": {"error": "episode not found", "seed": seed, "task_id": task_id},
            }

        score = final_ep.final_score

        # Clean up episode after evaluation
        del self._episodes[episode_id]

        return {
            "task_name": "qqr",
            "score": score / 100.0,
            "success": score >= 60,
            "time_taken": (datetime.now() - start_time).total_seconds(),
            "extra": {
                "conversation": final_ep.conversation,
                "tool_trace": final_ep.tool_trace,
                "seed": seed,
                "task_id": task_id,
                "usage": total_usage,
                "score_raw": score,
                "score_breakdown": final_ep.score_breakdown,
                "problem": final_ep.problem.to_dict(),
                "step_rewards": final_ep.step_rewards,
                "avg_step_reward": (
                    sum(final_ep.step_rewards) / len(final_ep.step_rewards)
                    if final_ep.step_rewards else 0.0
                ),
            },
        }

    async def _call_llm(
        self,
        client: httpx.AsyncClient,
        messages: List[Dict],
        model: str,
        base_url: str,
        api_key: str,
        temperature: float,
        timeout: int,
        tools: Optional[List[Dict]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Call LLM API with OpenAI Function Calling support."""
        try:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 4096,
            }

            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"

            print(f"[DEBUG] Sending messages count: {len(messages)}")
            # Print messages that contain tool_calls or are tool role
            for i, msg in enumerate(messages):
                role = msg.get('role')
                if role == 'tool' or 'tool_calls' in msg:
                    print(f"[DEBUG] msg[{i}] = {json.dumps(msg, ensure_ascii=False)[:300]}")

            resp = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=float(timeout),
            )

            if resp.status_code != 200:
                print(f"[DEBUG] LLM API Error: {resp.status_code} - {resp.text[:500]}")
                return None
            print(f"[DEBUG] LLM API success, parsing response")

            data = resp.json()
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})

            return {
                "content": message.get("content", ""),
                "tool_calls": message.get("tool_calls"),
                "usage": data.get("usage", {}),
            }

        except Exception as e:
            import traceback
            print(f"[DEBUG] LLM API Exception: {e}")
            traceback.print_exc()
            return None

    def _format_tool_results(self, results: List[Dict]) -> str:
        """Format tool return results."""
        return "\n\n".join(
            f"[{r['tool']}] 结果:\n{r['result']}"
            for r in results
        )

    async def cleanup(self):
        """Clean up resources."""
        # MCPState is singleton, usually doesn't need cleanup
        # But cleanup logic can be added here if needed
        pass
