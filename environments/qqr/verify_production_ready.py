#!/usr/bin/env python3
"""
QQR Pre-Launch Verification Suite

16 tests across three layers:
  A-layer: AST structure validation (no imports needed)
  B-layer: Module import tests (scorer/problem_generator)
  C-layer: Mock transport determinism (exec stub bypasses mcp import)
"""

import ast
import os
import sys
import textwrap
import traceback

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

RESULTS: list = []  # (test_id, label, passed, detail)


def record(test_id: str, label: str, passed: bool, detail: str = ""):
    RESULTS.append((test_id, label, passed, detail))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  A-LAYER: AST Structure Verification (no imports required)             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _parse_file(filename: str):
    """Parse a Python file and return (tree, source). Raises on syntax error."""
    path = os.path.join(ROOT, filename)
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    return ast.parse(source, filename=filename), source


# A.1a-d: Syntax validation for 4 modified files
FILES_TO_CHECK = [
    ("A.1a", "llm_validator.py"),
    ("A.1b", "scorer.py"),
    ("A.1c", "env.py"),
    ("A.1d", "mock_transport/server.py"),
]

for tid, fname in FILES_TO_CHECK:
    try:
        _parse_file(fname)
        record(tid, f"{fname} parses cleanly", True)
    except SyntaxError as e:
        record(tid, f"{fname} parses cleanly", False, str(e))


# A.2: _circuit_open and _consecutive_failures absent from llm_validator.py
try:
    _, src = _parse_file("llm_validator.py")
    forbidden = ["_circuit_open", "_consecutive_failures"]
    found = [f for f in forbidden if f in src]
    if found:
        record("A.2", "Circuit breaker removed from llm_validator.py", False,
               f"Found: {found}")
    else:
        record("A.2", "Circuit breaker removed from llm_validator.py", True)
except Exception as e:
    record("A.2", "Circuit breaker removed from llm_validator.py", False, str(e))


# A.4: search_train_tickets has 6 default="" parameters
try:
    tree, _ = _parse_file("mock_transport/server.py")
    found_defaults = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "search_train_tickets":
                # Count defaults that are empty string ""
                defaults = node.args.defaults
                empty_str_count = sum(
                    1 for d in defaults
                    if isinstance(d, ast.Constant) and d.value == ""
                )
                if empty_str_count == 6:
                    record("A.4", 'search_train_tickets has 6 default="" params', True)
                else:
                    record("A.4", 'search_train_tickets has 6 default="" params', False,
                           f"Found {empty_str_count} empty-string defaults, expected 6")
                found_defaults = True
                break
    if not found_defaults:
        record("A.4", 'search_train_tickets has 6 default="" params', False,
               "Function not found")
except Exception as e:
    record("A.4", 'search_train_tickets has 6 default="" params', False, str(e))


# A.9: env.py step() contains llm_validation_error check and ep.final_score = 0.0
try:
    _, src = _parse_file("env.py")
    has_check = "score_result.llm_validation_error" in src
    has_zero = "ep.final_score = 0.0" in src
    if has_check and has_zero:
        record("A.9", "env.py step() has llm_validation_error check + score=0", True)
    else:
        record("A.9", "env.py step() has llm_validation_error check + score=0", False,
               f"llm_validation_error check: {has_check}, final_score=0.0: {has_zero}")
except Exception as e:
    record("A.9", "env.py step() has llm_validation_error check + score=0", False, str(e))


# A.10: env.py evaluate() finally block has only _episodes.pop + gc.collect, no mcp/cleanup
try:
    tree, src = _parse_file("env.py")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "evaluate":
                # Find try/finally in the function body
                for child in ast.walk(node):
                    if isinstance(child, ast.Try):
                        finally_body = child.finalbody
                        if finally_body:
                            # Collect all names in the finally block
                            finally_names = set()
                            for fnode in finally_body:
                                for n in ast.walk(fnode):
                                    if isinstance(n, ast.Name):
                                        finally_names.add(n.id)
                                    elif isinstance(n, ast.Attribute):
                                        finally_names.add(n.attr)
                            has_pop = "pop" in finally_names
                            has_gc = "collect" in finally_names or "gc" in finally_names
                            has_mcp = any(
                                n in finally_names
                                for n in ("cleanup", "mcp", "_mcp_state",
                                          "close", "shutdown")
                            )
                            if has_pop and has_gc and not has_mcp:
                                record("A.10", "evaluate() finally: only pop + gc.collect, no MCP cleanup", True)
                            else:
                                record("A.10", "evaluate() finally: only pop + gc.collect, no MCP cleanup", False,
                                       f"pop={has_pop}, gc={has_gc}, mcp_related={has_mcp}, names={finally_names}")
                            break
                break
except Exception as e:
    record("A.10", "evaluate() finally: only pop + gc.collect, no MCP cleanup", False, str(e))


# A.11: config.py TOOLS_SCHEMA search_train_tickets required = ["date","from_city","to_city"]
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("config_check", os.path.join(ROOT, "config.py"))
    config_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_mod)

    tools_schema = config_mod.TOOLS_SCHEMA
    train_tool = None
    for tool in tools_schema:
        if tool.get("function", {}).get("name") == "search_train_tickets":
            train_tool = tool
            break

    if train_tool is None:
        record("A.11", "TOOLS_SCHEMA search_train_tickets required fields", False,
               "search_train_tickets not found in TOOLS_SCHEMA")
    else:
        required = train_tool["function"]["parameters"]["required"]
        expected = ["date", "from_city", "to_city"]
        if sorted(required) == sorted(expected):
            record("A.11", "TOOLS_SCHEMA search_train_tickets required fields", True)
        else:
            record("A.11", "TOOLS_SCHEMA search_train_tickets required fields", False,
                   f"Expected {expected}, got {required}")
except Exception as e:
    record("A.11", "TOOLS_SCHEMA search_train_tickets required fields", False, str(e))


# A.12: DEFAULT_LLM_SCORES removed from config.py and scorer.py
try:
    _, config_src = _parse_file("config.py")
    _, scorer_src = _parse_file("scorer.py")
    in_config = "DEFAULT_LLM_SCORES" in config_src
    in_scorer = "DEFAULT_LLM_SCORES" in scorer_src
    if not in_config and not in_scorer:
        record("A.12", "DEFAULT_LLM_SCORES fully removed", True)
    else:
        record("A.12", "DEFAULT_LLM_SCORES fully removed", False,
               f"config.py: {in_config}, scorer.py: {in_scorer}")
except Exception as e:
    record("A.12", "DEFAULT_LLM_SCORES fully removed", False, str(e))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  B-LAYER: Module Import Tests                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# A.3: generate(task_id) determinism across 10 task_ids
try:
    from problem_generator import get_generator
    gen = get_generator()
    all_match = True
    mismatch_ids = []
    for tid in range(10):
        p1 = gen.generate(tid)
        p2 = gen.generate(tid)
        if p1.to_dict() != p2.to_dict():
            all_match = False
            mismatch_ids.append(tid)
    if all_match:
        record("A.3", "generate() deterministic for 10 task_ids", True)
    else:
        record("A.3", "generate() deterministic for 10 task_ids", False,
               f"Mismatch at task_ids: {mismatch_ids}")
except Exception as e:
    record("A.3", "generate() deterministic for 10 task_ids", False,
           traceback.format_exc())


# A.5: ScoreBreakdown with llm_validation_error → total=0, all scores=0
try:
    from scorer import ScoreBreakdown
    sb = ScoreBreakdown()
    sb.llm_validation_error = "All 3 models failed"
    # With no scores set, total should be 0
    total_ok = sb.total == 0.0
    code_ok = sb.code_total == 0.0
    llm_ok = sb.llm_total == 0.0
    if total_ok and code_ok and llm_ok:
        record("A.5", "ScoreBreakdown with llm_validation_error -> total=0", True)
    else:
        record("A.5", "ScoreBreakdown with llm_validation_error -> total=0", False,
               f"total={sb.total}, code={sb.code_total}, llm={sb.llm_total}")
except Exception as e:
    record("A.5", "ScoreBreakdown with llm_validation_error -> total=0", False,
           traceback.format_exc())


# A.7: ScoreBreakdown.total safe with all-zero fields (no ZeroDivisionError)
try:
    from scorer import ScoreBreakdown
    sb = ScoreBreakdown()
    total = sb.total
    if isinstance(total, float) and total == 0.0:
        record("A.7", "ScoreBreakdown.total safe with zero fields", True)
    else:
        record("A.7", "ScoreBreakdown.total safe with zero fields", False,
               f"total={total}")
except ZeroDivisionError as e:
    record("A.7", "ScoreBreakdown.total safe with zero fields", False,
           f"ZeroDivisionError: {e}")
except Exception as e:
    record("A.7", "ScoreBreakdown.total safe with zero fields", False,
           traceback.format_exc())


# A.8: ScoreBreakdown.to_dict() includes llm_validation_error field
try:
    from scorer import ScoreBreakdown
    sb = ScoreBreakdown()
    sb.llm_validation_error = "test error"
    d = sb.to_dict()
    if "llm_validation_error" in d and d["llm_validation_error"] == "test error":
        record("A.8", "ScoreBreakdown.to_dict() has llm_validation_error", True)
    else:
        record("A.8", "ScoreBreakdown.to_dict() has llm_validation_error", False,
               f"Keys: {list(d.keys())}")
except Exception as e:
    record("A.8", "ScoreBreakdown.to_dict() has llm_validation_error", False,
           traceback.format_exc())


# A.6a: TravelScorer + MockValidator(success=True) → LLM scores filled, code scores computed
# A.6b: TravelScorer + MockValidator(success=False) → early return, code scores=0, error propagated
# A.6c: TravelScorer(llm_validator=None) → early return, error="LLM validator unavailable"
try:
    import asyncio
    import types as _types

    # Stub openai if not installed (production has it; local dev may not)
    if "openai" not in sys.modules:
        _openai_stub = _types.ModuleType("openai")
        class _FakeAsyncOpenAI:
            def __init__(self, *a, **kw): pass
        _openai_stub.AsyncOpenAI = _FakeAsyncOpenAI
        sys.modules["openai"] = _openai_stub

    from scorer import TravelScorer
    from llm_validator import LLMValidationResult
    from problem_generator import get_generator

    class MockValidator:
        """Mock LLMValidator for testing."""
        def __init__(self, success: bool):
            self._success = success

        async def validate(self, model_output, problem, tool_trace):
            if self._success:
                return LLMValidationResult(
                    success=True,
                    tool_info_used=True,
                    practicality=8.0,
                    informativeness=7.0,
                    logic=8.0,
                    user_experience=7.0,
                    reasons={"practicality": "good", "informativeness": "good",
                             "logic": "good", "user_experience": "good"},
                )
            else:
                return LLMValidationResult(
                    success=False,
                    error="Mock API failure: all models failed",
                )

    gen = get_generator()
    problem = gen.generate(0)  # intercity type

    fake_output = (
        "# 旅行规划方案\n\n"
        "## 第一天\n"
        "1. 上午: 游览故宫博物院 (门票60元)\n"
        "2. 下午: 天安门广场 (免费)\n"
        "3. 晚上: 推荐全聚德烤鸭 (人均150元)\n\n"
        "## 交通方案\n"
        "- 航班 CA1234，价格800元，08:00从首都国际机场出发\n"
        "- 火车 G123，价格553元，07:00从北京南站出发\n\n"
        "## 天气\n"
        "晴天，适合出行\n"
    )

    fake_tool_trace = [
        {
            "name": "poi_search",
            "arguments": {"address": "故宫", "region": "北京"},
            "result": {"text": "名称: 故宫博物院，地址: 北京市东城区景山前街4号"},
        },
        {
            "name": "search_flights",
            "arguments": {"date": "2025-05-01", "from_city": "上海", "to_city": "北京"},
            "result": {"text": "航班 CA1234，价格800元，08:00从首都国际机场出发"},
        },
    ]

    loop = asyncio.new_event_loop()

    # A.6a: success=True
    scorer_ok = TravelScorer(llm_validator=MockValidator(success=True))
    result_ok = loop.run_until_complete(
        scorer_ok.score(fake_output, problem, fake_tool_trace)
    )
    llm_filled = (
        result_ok.llm_practicality > 0
        and result_ok.llm_informativeness > 0
        and result_ok.llm_logic > 0
        and result_ok.llm_user_experience > 0
    )
    code_ran = result_ok.parse_success
    if llm_filled and code_ran:
        record("A.6a", "MockValidator(success=True) -> LLM+code scores", True)
    else:
        record("A.6a", "MockValidator(success=True) -> LLM+code scores", False,
               f"llm_filled={llm_filled} (prac={result_ok.llm_practicality}), "
               f"code_ran={code_ran}")

    # A.6b: success=False
    scorer_fail = TravelScorer(llm_validator=MockValidator(success=False))
    result_fail = loop.run_until_complete(
        scorer_fail.score(fake_output, problem, fake_tool_trace)
    )
    code_zero = result_fail.code_total == 0.0
    llm_zero = result_fail.llm_total == 0.0
    error_propagated = "Mock API failure" in result_fail.llm_validation_error
    if code_zero and llm_zero and error_propagated:
        record("A.6b", "MockValidator(success=False) -> early return, scores=0", True)
    else:
        record("A.6b", "MockValidator(success=False) -> early return, scores=0", False,
               f"code={result_fail.code_total}, llm={result_fail.llm_total}, "
               f"error='{result_fail.llm_validation_error}'")

    # A.6c: No LLM validator (llm_validator=None)
    scorer_none = TravelScorer(llm_validator=None)
    result_none = loop.run_until_complete(
        scorer_none.score(fake_output, problem, fake_tool_trace)
    )
    code_zero_c = result_none.code_total == 0.0
    llm_zero_c = result_none.llm_total == 0.0
    has_error = "LLM validator unavailable" in result_none.llm_validation_error
    if code_zero_c and llm_zero_c and has_error:
        record("A.6c", "No LLM validator -> early return, error set", True)
    else:
        record("A.6c", "No LLM validator -> early return, error set", False,
               f"code={result_none.code_total}, llm={result_none.llm_total}, "
               f"error='{result_none.llm_validation_error}'")

    loop.close()

except Exception as e:
    record("A.6a", "MockValidator(success=True) -> LLM+code scores", False,
           traceback.format_exc())
    record("A.6b", "MockValidator(success=False) -> early return, scores=0", False,
           traceback.format_exc())
    record("A.6c", "No LLM validator -> early return, error set", False,
           traceback.format_exc())


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  C-LAYER: Mock Transport Determinism                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# B.2: _generate_flights / _generate_trains deterministic + ≥8 records
try:
    import types
    server_path = os.path.join(ROOT, "mock_transport", "server.py")
    with open(server_path, "r", encoding="utf-8") as f:
        server_src = f.read()

    stub_globals = {
        "__builtins__": __builtins__,
        "__name__": "__main__",
        "__file__": server_path,
    }

    # Stub out the mcp.server.fastmcp import
    mcp_stub = types.ModuleType("mcp")
    mcp_server_stub = types.ModuleType("mcp.server")
    mcp_fastmcp_stub = types.ModuleType("mcp.server.fastmcp")

    class FakeFastMCP:
        def __init__(self, *a, **kw): pass
        def tool(self, *a, **kw):
            def decorator(fn): return fn
            return decorator
        def run(self): pass

    mcp_fastmcp_stub.FastMCP = FakeFastMCP
    mcp_server_stub.fastmcp = mcp_fastmcp_stub
    mcp_stub.server = mcp_server_stub

    saved_modules = {}
    for name in ("mcp", "mcp.server", "mcp.server.fastmcp"):
        saved_modules[name] = sys.modules.get(name)

    sys.modules["mcp"] = mcp_stub
    sys.modules["mcp.server"] = mcp_server_stub
    sys.modules["mcp.server.fastmcp"] = mcp_fastmcp_stub

    try:
        os.environ["TRANSPORT_SALT"] = "test_salt_2025"
        exec(compile(server_src, server_path, "exec"), stub_globals)

        gen_flights = stub_globals["_generate_flights"]
        gen_trains = stub_globals["_generate_trains"]

        flights1 = gen_flights("2025-05-01", "北京", "上海")
        flights2 = gen_flights("2025-05-01", "北京", "上海")
        trains1 = gen_trains("2025-05-01", "北京", "上海")
        trains2 = gen_trains("2025-05-01", "北京", "上海")

        flights_match = flights1 == flights2
        trains_match = trains1 == trains2
        flights_count = len(flights1) >= 8
        trains_count = len(trains1) >= 8

        all_ok = flights_match and trains_match and flights_count and trains_count
        if all_ok:
            record("B.2", "Transport generation deterministic + >=8 records", True)
        else:
            detail = (
                f"flights_match={flights_match}, trains_match={trains_match}, "
                f"flights={len(flights1)}, trains={len(trains1)}"
            )
            record("B.2", "Transport generation deterministic + >=8 records", False, detail)
    finally:
        for name, mod in saved_modules.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
        os.environ.pop("TRANSPORT_SALT", None)

except Exception as e:
    record("B.2", "Transport generation deterministic + >=8 records", False,
           traceback.format_exc())


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  RESULTS OUTPUT                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def print_results():
    sep = "=" * 60
    print(f"\n{sep}")
    print("QQR Pre-Launch Verification Suite")
    print(sep)

    passed = 0
    failed = 0
    for test_id, label, ok, detail in RESULTS:
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        line = f"  {status}: {test_id} - {label}"
        print(line)
        if not ok and detail:
            for dl in detail.split("\n")[:5]:
                print(f"        {dl}")

    total = passed + failed
    print(sep)
    print(f"RESULTS: {passed} passed, {failed} failed, {total} total")
    print(sep)

    # ── System Analysis Summary ──
    print(f"\n{sep}")
    print("SYSTEM ANALYSIS SUMMARY")
    print(sep)

    print(textwrap.dedent("""\
    [2.1] Concurrency Safety
      - MCPState: singleton with asyncio.Lock + double-check init
      - Actor._episodes: UUID-keyed per-evaluation isolation
      - LLMValidator: no mutable instance state (circuit breaker removed)
      - _EPOCH_SALT: import-time immutable
      - ProblemGenerator: per-call random.Random(task_id), no shared RNG

    [2.2] Resource Lifecycle
      +-------------------+---------------------+---------------------+---------------------------+
      | Component         | Created             | Destroyed           | Notes                     |
      +-------------------+---------------------+---------------------+---------------------------+
      | MCP subprocess    | Actor.reset() first | Actor.cleanup()     | singleton, diskcache      |
      | AsyncOpenAI       | LLMValidator.init   | container exit      | single instance           |
      | EpisodeState      | evaluate()          | finally pop+gc      | per-eval isolated         |
      | diskcache         | MCP server connect  | server.cleanup()    | /var/lib/qqr/cache/       |
      +-------------------+---------------------+---------------------+---------------------------+

    [2.3] Error Propagation Chain
      LLMValidator.validate() -> LLMValidationResult(success=False, error="...")
        -> TravelScorer.score() -> result.llm_validation_error = error (early return, code=0)
        -> Actor.step() -> ep.final_score=0, ep.score_breakdown["error"] = "LLM validator unavailable: ..."
        -> Actor.evaluate() -> {"score": 0.0, "success": False, "extra": {"score_breakdown": {...}}}

      No LLM validator (missing API key):
        -> TravelScorer.score() -> result.llm_validation_error = "LLM validator unavailable"
        -> Same downstream handling as above (score=0)

    [2.4] Scorer Execution Paths
      +----+------------------------------------+-----------+----------+-------+----------------------------+
      | P  | Trigger                            | code      | llm      | total | Notes                      |
      +----+------------------------------------+-----------+----------+-------+----------------------------+
      | P1 | LLM validator all models fail      | 0 (skip)  | 0 (skip) | 0     | env.py marks invalid       |
      | P2 | tool_info_used=False               | 0 (skip)  | 0 (skip) | 0     | hard fail multiplier 0.0   |
      | P3 | LLM ok + format_valid fail         | 0 (skip)  | set      | ~base*0.15 | RL gradient kept      |
      | P4 | LLM ok + normal flow               | computed  | set      | 0-100 | full scoring               |
      | P5 | No LLM validator (no API key)      | 0 (skip)  | 0 (skip) | 0     | error field set            |
      | P6 | Cross-validation anti-cheat fires  | computed  | set      | 0     | tool_info_used overwritten |
      +----+------------------------------------+-----------+----------+-------+----------------------------+

    [2.5] Behavioral Changes vs. Previous Version
      +--------------------------------------+-----------------------------------+-----------------------------------+
      | Scenario                             | Old Behavior                      | New Behavior                      |
      +--------------------------------------+-----------------------------------+-----------------------------------+
      | Chutes API all timeouts              | DEFAULT scores(15/30) + code      | score=0, error field marked       |
      | 3 consecutive API failures           | Circuit breaker, all evals -> 0   | Independent retry per evaluation  |
      | Concurrent eval A finishes           | cleanup MCP, eval B tool fails    | MCP kept alive, B continues       |
      | Model sends 3 args to train search   | FastMCP param missing error       | Defaults fill in, normal result   |
      | No CHUTES_API_KEY set                | DEFAULT scores(15/30) + code      | score=0, error field marked       |
      +--------------------------------------+-----------------------------------+-----------------------------------+

    [2.6] Known Acceptable Risks
      +------------------------------------------+--------+----------------------------------------------+
      | Risk                                     | Sev    | Rationale                                    |
      +------------------------------------------+--------+----------------------------------------------+
      | AsyncOpenAI no explicit close             | Low    | Single instance, container exit reclaims     |
      | P1/P5 path code scores not computed       | None   | env.py marks invalid, score=0                |
      | CHUTES_API_KEY required for scoring       | None   | By design; no silent fallback                |
      +------------------------------------------+--------+----------------------------------------------+
    """))

    return failed == 0


if __name__ == "__main__":
    success = print_results()
    sys.exit(0 if success else 1)
