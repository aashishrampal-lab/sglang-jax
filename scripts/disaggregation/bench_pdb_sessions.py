#!/usr/bin/env python3
"""N-Session Continuous Closed-Loop Benchmark for GLM-5.2 (Colocated & PD Disagg).

Features:
  1. Empirical Prefill Length (ISL) Distribution:
     - 1-8K   : 50%
     - 8-16K  : 20%
     - 16-32K : 12%
     - 32-64K :  9%
     - 64K+   :  9% (64K-92K, fitting safely inside context length)
  2. Authentic Multi-Turn Messages, Tools & Original max_tokens from requests_mocked.jsonl:
     - Preserves top-level `tools` (and `tool_choice`) and structured multi-turn
       `messages` (`system`, `user`, `assistant`, `tool`, `tool_calls`) from
       `requests_mocked.jsonl`.
     - Uses each request's original `max_tokens` from `requests_mocked.jsonl`
       by default (`--max-tokens 0`), and sets `ignore_eos = False` by default
       (`--no-ignore-eos`) so generation stops naturally on EOS / tool call / turn stop.
  3. N Separate Cache Requests in Phase 0 (80% Prefix Cache Hit Rate):
     - Instead of pre-populating shared 92K MasterTrunks, Phase 0 flushes the cache
       and warms `n` separate requests (one distinct prefix per session `s` in `0..n-1`,
       sized to each session's own `target_isl_tokens`).
     - In Phase 1 (measured benchmark), each of the `n` concurrent sessions reuses its
       own warmed 80% prefix (`tools` + `prefix_messages`) and varies the 20% uncached
       suffix (`suffix_messages`) on each turn.
  4. Balanced Interleaved Per-DP Queue (No Head-of-Line Blocking):
     - Stratified specs are balanced across `dp_rank` and shuffled within each DP rank
       so Wave 1 has the natural 50%/20%/12%/9%/9% bucket mix.
  5. Continuous Closed-Loop Steady-State Window (--duration-per-n):
     - Each of the `n` sessions runs continuously in a closed loop for `--duration-per-n`
       seconds (or `--turns-per-session` turns when `--duration-per-n == 0`).
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import lzma
from pathlib import Path
import random
import statistics
import time
from typing import Any
import aiohttp


CHARS_PER_TOKEN = 3.85

ISL_BUCKETS = [
    ("1-8K", 0.50, 5376, 8064),
    ("8-16K", 0.20, 8320, 15872),
    ("16-32K", 0.12, 16384, 31744),
    ("32-64K", 0.09, 32768, 63488),
    ("64K+", 0.09, 65536, 92160),
]


def _extract_text(content: Any) -> str:
  if isinstance(content, str):
    return content
  if isinstance(content, list):
    parts = []
    for p in content:
      if isinstance(p, dict) and isinstance(p.get("text"), str):
        parts.append(p["text"])
      else:
        parts.append(json.dumps(p, ensure_ascii=False))
    return "\n".join(parts)
  return str(content or "")


def _msg_char_len(m: dict[str, Any]) -> int:
  c = len(_extract_text(m.get("content"))) + len(str(m.get("reasoning_content") or ""))
  if m.get("tool_calls"):
    c += len(json.dumps(m["tool_calls"], ensure_ascii=False))
  return max(16, c + 16)


def load_raw_requests(path: str) -> list[dict[str, Any]]:
  """Loads requests_mocked.jsonl(.xz) preserving structured messages, tools, and max_tokens."""
  p = Path(path)
  if p.suffix == ".xz":
    with lzma.open(p, "rt", encoding="utf-8") as f:
      raw_lines = [line.strip() for line in f if line.strip()]
  else:
    with open(p, "r", encoding="utf-8") as f:
      raw_lines = [line.strip() for line in f if line.strip()]

  raw_reqs: list[dict[str, Any]] = []
  for line in raw_lines:
    obj = json.loads(line)
    raw_reqs.append(obj.get("body", obj))
  # Prioritize multi-turn tool-calling requests with bounded max_tokens (excluding Row 0's
  # synthetic 1-word "What is 1+1?" prompt and 384k-token runaway rows) so all 5 ISL buckets
  # exercise real structured tools + multi-turn messages with balanced output lengths.
  tool_bounded = [
      r for r in raw_reqs if r.get("tools") and int(r.get("max_tokens") or 0) <= 32768
  ]
  if tool_bounded:
    return tool_bounded
  tool_any = [r for r in raw_reqs if r.get("tools")]
  return tool_any if tool_any else (raw_reqs[1:] if len(raw_reqs) > 1 else raw_reqs)


def build_bucket_size_specs(dp_size: int = 1) -> dict[int, list[dict[str, Any]]]:
  """Builds canonical size specs for the 5 ISL buckets per DP rank so requests of the same size share the same 80% prefill cache."""
  per_dp_buckets: dict[int, list[dict[str, Any]]] = {}
  for r in range(dp_size):
    b_list: list[dict[str, Any]] = []
    for b_idx, (bucket_name, weight, lo_tok, hi_tok) in enumerate(ISL_BUCKETS):
      target_tok = (((lo_tok + hi_tok) // 2) // 128) * 128
      b_list.append({
          "bucket_idx": b_idx,
          "bucket": bucket_name,
          "weight": weight,
          "target_isl_tokens": target_tok,
          "dp_rank": r,
      })
    per_dp_buckets[r] = b_list
  return per_dp_buckets


def build_session_chat_payload(
    orig_req: dict[str, Any],
    *,
    num_sessions: int,
    session_id: int,
    dp_rank: int,
    target_isl_tokens: int,
    cache_hit_ratio: float,
    uid: str,
    max_tokens: int,
    ignore_eos: bool,
    is_warmup: bool = False,
) -> dict[str, Any]:
  """Builds a structured multi-turn + tools payload where requests of the same `target_isl_tokens` size share the same 80% prefill cache."""
  total_chars = max(1024, int(target_isl_tokens * CHARS_PER_TOKEN))
  prefix_budget = int(total_chars * cache_hit_ratio)
  suffix_budget = max(256, total_chars - prefix_budget)

  # Size-based prefix tag placed at the very start of the prompt (in tools[0] if tools exist,
  # else in messages[0]) so all requests of the same `target_isl_tokens` size reuse the same
  # 80% prefill cache in RadixCache.
  size_tag = f"[size_cache_key=isl{target_isl_tokens:06d}_dp{dp_rank:02d}] "
  raw_tools = copy.deepcopy(orig_req.get("tools") or [])

  kept_tools: list[dict[str, Any]] = []
  tools_chars = 0
  max_tools_chars = max(0, prefix_budget - 512)
  for t in raw_tools:
    tc_len = len(json.dumps(t, ensure_ascii=False)) + 2
    if tools_chars + tc_len <= max_tools_chars or (not kept_tools and tc_len <= max_tools_chars):
      kept_tools.append(t)
      tools_chars += tc_len
    else:
      break

  orig_msgs = copy.deepcopy(orig_req.get("messages") or [])
  if kept_tools:
    if "function" in kept_tools[0] and isinstance(kept_tools[0]["function"], dict):
      desc = str(kept_tools[0]["function"].get("description") or "")
      kept_tools[0]["function"]["description"] = size_tag + desc
  else:
    if orig_msgs and isinstance(orig_msgs[0].get("content"), str):
      orig_msgs[0]["content"] = size_tag + orig_msgs[0]["content"]
    else:
      orig_msgs.insert(0, {"role": "system", "content": size_tag})

  # Build structured `prefix_messages` up to `rem_prefix` chars (identical between Warmup and Phase 1 turns of session `s`)
  rem_prefix = max(256, prefix_budget - tools_chars)
  prefix_messages: list[dict[str, Any]] = []
  cur_p = 0
  rep = 0
  while cur_p < rem_prefix:
    for m in orig_msgs:
      m_copy = copy.deepcopy(m)
      mlen = _msg_char_len(m_copy)
      if cur_p + mlen <= rem_prefix:
        prefix_messages.append(m_copy)
        cur_p += mlen
      else:
        need = rem_prefix - cur_p
        if need >= 32:
          txt = _extract_text(m_copy.get("content"))
          if len(txt) < need:
            pad_unit = txt + "\n" + json.dumps(m_copy, ensure_ascii=False)
            txt = pad_unit * ((need // max(1, len(pad_unit))) + 2)
          m_slice: dict[str, Any] = {
              "role": m_copy.get("role", "user"),
              "content": txt[:need],
          }
          if m_slice["role"] == "tool" and m_copy.get("tool_call_id"):
            m_slice["tool_call_id"] = m_copy["tool_call_id"]
          prefix_messages.append(m_slice)
          cur_p += need
        cur_p = rem_prefix
        break
    rep += 1
    if rep > 50:
      break

  # Build `suffix_messages` (20% uncached tail) starting with unique per-turn `uid`
  turn_tag = f"[turn_uid=s{session_id:04d}_{uid}]\n"
  tail_texts: list[str] = []
  cur_s = len(turn_tag)
  while cur_s < suffix_budget:
    for m in reversed(orig_msgs):
      txt = _extract_text(m.get("content"))
      if txt.strip():
        tail_texts.append(txt)
        cur_s += len(txt) + 2
        if cur_s >= suffix_budget:
          break
    if not tail_texts:
      tail_texts.append("Continue the analysis and call the appropriate tool if needed.")
      break
  suffix_str = (turn_tag + "\n".join(tail_texts))[:suffix_budget]
  suffix_messages = [{"role": "user", "content": suffix_str}]

  # Use original request's `max_tokens` from requests_mocked.jsonl when `max_tokens <= 0`
  orig_max_tokens = int(orig_req.get("max_tokens") or 8192)
  req_max_tokens = max_tokens if max_tokens > 0 else orig_max_tokens
  req_ignore_eos = ignore_eos
  # Override 1-8K bucket to output length 1024 with ignore_eos=False (32-64K override removed)
  if target_isl_tokens < 8192:
    req_max_tokens = 1024
  if is_warmup:
    req_max_tokens = 1
    req_ignore_eos = False

  payload: dict[str, Any] = {
      "model": "/weights",
      "messages": prefix_messages + suffix_messages,
      "chat_template_kwargs": {
          "enable_thinking": True,
          "reasoning_effort": "high",
          "clear_thinking": False,
      },
      "max_tokens": req_max_tokens,
      "temperature": 1.0,
      "ignore_eos": req_ignore_eos,
      "stream": True,
      "stream_options": {"include_usage": True},
      "dp_rank": int(dp_rank),
      "rid": f"s{session_id}_dp{dp_rank}_{uid}",
  }
  if kept_tools:
    payload["tools"] = kept_tools
    if orig_req.get("tool_choice") is not None:
      payload["tool_choice"] = orig_req["tool_choice"]
  return payload


async def send_chat_request(
    http_session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, Any]:
  t_start = time.perf_counter()
  ttft: float | None = None
  chunk_times: list[float] = []
  prompt_tokens = 0
  completion_tokens = 0
  cached_tokens = 0
  status_code = 0
  error_msg: str | None = None
  num_content_chunks = 0
  finish_reason: str | None = None
  has_tool_calls = False

  headers = {
      "X-SMG-Routing-Key": f"session-{meta.get('session_id', 0)}",
      "X-Session-ID": f"session-{meta.get('session_id', 0)}",
  }

  try:
    async with http_session.post(url, json=payload, headers=headers) as resp:
      status_code = resp.status
      if status_code != 200:
        err_body = await resp.text()
        error_msg = f"HTTP {status_code}: {err_body[:300]}"
      else:
        async for raw_line in resp.content:
          line = raw_line.decode("utf-8", errors="replace").strip()
          if not line or not line.startswith("data:"):
            continue
          data_str = line[5:].strip()
          if data_str == "[DONE]":
            break
          try:
            chunk = json.loads(data_str)
          except Exception:
            continue

          now = time.perf_counter()
          choices = chunk.get("choices")
          if choices and len(choices) > 0:
            c0 = choices[0]
            if c0.get("finish_reason"):
              finish_reason = c0.get("finish_reason")
            delta = c0.get("delta", {})
            text_val = c0.get("text", "")
            if delta.get("tool_calls"):
              has_tool_calls = True
            has_token = bool(
                delta.get("content")
                or delta.get("reasoning_content")
                or delta.get("tool_calls")
                or text_val
            )
            if has_token:
              if ttft is None:
                ttft = now - t_start
              chunk_times.append(now)
              num_content_chunks += 1

          usage = chunk.get("usage")
          if isinstance(usage, dict) and usage:
            prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
            completion_tokens = int(usage.get("completion_tokens") or completion_tokens)
            details = usage.get("prompt_tokens_details")
            if isinstance(details, dict) and details.get("cached_tokens") is not None:
              cached_tokens = int(details["cached_tokens"])
  except Exception as e:
    error_msg = f"{type(e).__name__}: {e}"

  t_end = time.perf_counter()
  e2e = t_end - t_start
  if ttft is None and error_msg is None:
    ttft = e2e

  eff_compl = completion_tokens or num_content_chunks
  decode_dur = max(0.0, e2e - (ttft or 0.0))
  per_token_itl: float | None = None
  decode_tps: float | None = None
  if eff_compl > 1 and decode_dur > 0:
    per_token_itl = decode_dur / (eff_compl - 1)
    decode_tps = (eff_compl - 1) / decode_dur

  chunk_itls: list[float] = []
  if len(chunk_times) >= 2:
    chunk_itls = [chunk_times[i] - chunk_times[i - 1] for i in range(1, len(chunk_times))]
  elif per_token_itl is not None:
    chunk_itls = [per_token_itl]

  return {
      **meta,
      "t_start": t_start,
      "t_end": t_end,
      "ttft": ttft,
      "e2e": e2e,
      "decode_dur": decode_dur,
      "per_token_itl": per_token_itl,
      "decode_tps": decode_tps,
      "itls": chunk_itls,
      "chunk_times": chunk_times,
      "prompt_tokens": prompt_tokens,
      "completion_tokens": eff_compl,
      "cached_tokens": cached_tokens,
      "finish_reason": finish_reason,
      "has_tool_calls": has_tool_calls,
      "status_code": status_code,
      "error": error_msg,
  }


def percentile(vals: list[float], p: float) -> float:
  if not vals:
    return 0.0
  s = sorted(vals)
  k = (len(s) - 1) * (p / 100.0)
  f = int(k)
  c = min(f + 1, len(s) - 1)
  if f == c:
    return s[f]
  return s[f] + (s[c] - s[f]) * (k - f)


async def run_single_n(
    http_session: aiohttp.ClientSession,
    *,
    base_url: str,
    prefill_url: str,
    decode_url: str,
    target_url: str,
    raw_reqs: list[dict[str, Any]],
    num_sessions: int,
    turns_per_session: int,
    duration_per_n: float,
    cache_hit_ratio: float,
    max_tokens: int,
    ignore_eos: bool,
    dp_size: int,
    flush_before_warmup: bool,
    warmup_concurrency: int = 16,
) -> dict[str, Any]:
  per_dp_buckets = build_bucket_size_specs(dp_size=dp_size)
  warmup_specs = [sp for r in range(dp_size) for sp in per_dp_buckets[r]]

  if flush_before_warmup:
    await asyncio.sleep(3.0)
    for u in {base_url, prefill_url, decode_url}:
      if not u:
        continue
      try:
        async with http_session.post(
            f"{u.rstrip('/')}/flush_cache", timeout=aiohttp.ClientTimeout(total=15)
        ) as _:
          pass
      except Exception:
        pass
    await asyncio.sleep(1.0)

  # Phase 0: Pre-populate the 5 canonical ISL size caches per DP rank (requests of the same size reuse the same prefill cache)
  total_warm_isl = sum(sp["target_isl_tokens"] for sp in warmup_specs)
  print(
      f"\n[n={num_sessions}] Phase 0 (Warmup - Excluded from Benchmark): "
      f"Pre-populating {len(warmup_specs)} size-based prefill caches across {dp_size} DP rank(s) "
      f"(total target ISL={total_warm_isl:,} tokens, 80% prefix={int(total_warm_isl * cache_hit_ratio):,} tokens)..."
  )
  t_w0 = time.perf_counter()
  warmup_sem = asyncio.Semaphore(max(1, min(warmup_concurrency, len(warmup_specs))))

  async def _warm_one_size(sp: dict[str, Any]) -> dict[str, Any]:
    b_idx = sp["bucket_idx"]
    dp_rank = sp["dp_rank"]
    target_isl = sp["target_isl_tokens"]
    orig_req = raw_reqs[(b_idx + dp_rank * 5) % len(raw_reqs)]
    async with warmup_sem:
      w_payload = build_session_chat_payload(
          orig_req,
          num_sessions=num_sessions,
          session_id=b_idx,
          dp_rank=dp_rank,
          target_isl_tokens=target_isl,
          cache_hit_ratio=cache_hit_ratio,
          uid=f"warmup_n{num_sessions}_dp{dp_rank}_b{b_idx}",
          max_tokens=1,
          ignore_eos=False,
          is_warmup=True,
      )
      return await send_chat_request(
          http_session,
          target_url,
          w_payload,
          {
              "session_id": b_idx,
              "dp_rank": dp_rank,
              "bucket": sp["bucket"],
              "target_isl": target_isl,
              "turn": -1,
          },
      )

  w_all = await asyncio.gather(*(_warm_one_size(sp) for sp in warmup_specs))
  w_ok = sum(1 for x in w_all if x["error"] is None)
  w_prompt_sum = sum(x["prompt_tokens"] for x in w_all if x["error"] is None)
  print(
      f"[n={num_sessions}] Phase 0 Warmup complete: {w_ok}/{len(warmup_specs)} size caches populated "
      f"({w_prompt_sum:,} total prompt tokens) in {time.perf_counter() - t_w0:.2f}s. Starting measured benchmark..."
  )
  await asyncio.sleep(1.0)

  # Phase 1: Measured n-Session Continuous Benchmark
  # Maintain target bucket distribution (50% 1-8K, 20% 8-16K, 12% 16-32K, 9% 32-64K, 9% 64K+)
  # at the server via deficit-steered probability sampling where requests of the same size reuse
  # the same 80% prefill cache.
  t_bench_start = time.perf_counter()
  t_window_end = (t_bench_start + duration_per_n) if duration_per_n > 0 else 0.0
  dist_rng = random.Random(42 + num_sessions * 97)
  active_in_flight = [0] * len(ISL_BUCKETS)
  dispatched_counts = [0] * len(ISL_BUCKETS)

  def sample_next_bucket_spec(dp_rank: int) -> dict[str, Any]:
    d_tot = sum(dispatched_counts)
    # Deterministically select the bucket with the largest dispatch deficit relative to
    # target probabilities (50% 1-8K, 20% 8-16K, 12% 16-32K, 9% 32-64K, 9% 64K+)
    # so the dispatched request counts strictly follow the 50:20:12:9:9 ratio.
    b_chosen = max(
        range(len(ISL_BUCKETS)),
        key=lambda b_idx: ((d_tot + 1) * ISL_BUCKETS[b_idx][1] - dispatched_counts[b_idx], ISL_BUCKETS[b_idx][1]),
    )
    active_in_flight[b_chosen] += 1
    dispatched_counts[b_chosen] += 1
    return per_dp_buckets[dp_rank][b_chosen]

  async def session_worker(s_id: int) -> list[dict[str, Any]]:
    dp_rank = s_id % dp_size
    out = []
    t = 0
    while True:
      if duration_per_n > 0:
        if time.perf_counter() >= t_window_end:
          break
      else:
        if t >= turns_per_session:
          break

      sp = sample_next_bucket_spec(dp_rank)
      b_idx = sp["bucket_idx"]
      bucket = sp["bucket"]
      target_isl = sp["target_isl_tokens"]
      orig_req = raw_reqs[(b_idx + dp_rank * 5) % len(raw_reqs)]
      uid = f"n{num_sessions}_s{s_id}_t{t}_{int(time.perf_counter() * 1000) % 1000000}"
      payload = build_session_chat_payload(
          orig_req,
          num_sessions=num_sessions,
          session_id=s_id,
          dp_rank=dp_rank,
          target_isl_tokens=target_isl,
          cache_hit_ratio=cache_hit_ratio,
          uid=uid,
          max_tokens=max_tokens,
          ignore_eos=ignore_eos,
          is_warmup=False,
      )
      try:
        res = await send_chat_request(
            http_session,
            target_url,
            payload,
            {
                "session_id": s_id,
                "dp_rank": dp_rank,
                "bucket": bucket,
                "target_isl": target_isl,
                "req_max_tokens": payload["max_tokens"],
                "num_tools": len(payload.get("tools") or []),
                "turn": t,
            },
        )
      finally:
        active_in_flight[b_idx] = max(0, active_in_flight[b_idx] - 1)
      out.append(res)
      t += 1
    return out

  worker_lists = await asyncio.gather(*(session_worker(s_id) for s_id in range(num_sessions)))
  t_bench_end = time.perf_counter()
  wall_clock = t_bench_end - t_bench_start
  eff_window_s = duration_per_n if duration_per_n > 0 else wall_clock

  results = [r for lst in worker_lists for r in lst]
  ok_results = [r for r in results if r["error"] is None]
  err_results = [r for r in results if r["error"] is not None]

  # Compute tokens streamed strictly inside [t_bench_start, t_window_end] (Active Steady-State Window)
  window_output_tokens = 0.0
  window_input_tokens = 0.0
  window_uncached_tokens = 0.0
  window_completed_reqs = 0
  for r in ok_results:
    c_times = r.get("chunk_times") or []
    compl = float(r["completion_tokens"] or 0)
    prompt = float(r["prompt_tokens"] or 0)
    cached = float(r["cached_tokens"] or 0)
    uncached = max(0.0, prompt - cached)
    if duration_per_n <= 0 or r["t_end"] <= t_window_end:
      window_output_tokens += compl
      window_input_tokens += prompt
      window_uncached_tokens += uncached
      window_completed_reqs += 1
    elif c_times:
      in_win_chunks = sum(1 for ct in c_times if ct <= t_window_end)
      if in_win_chunks > 0:
        frac = in_win_chunks / len(c_times)
        window_output_tokens += compl * frac
        window_input_tokens += prompt
        window_uncached_tokens += uncached

  win_ok_results = [r for r in ok_results if duration_per_n <= 0 or r["t_end"] <= t_window_end]
  eval_results = win_ok_results if len(win_ok_results) >= max(4, num_sessions // 2) else ok_results

  ttfts = [r["ttft"] for r in eval_results if r["ttft"] is not None]
  e2es = [r["e2e"] for r in eval_results if r["e2e"] is not None]
  per_tok_itls = [r["per_token_itl"] for r in eval_results if r.get("per_token_itl") is not None]
  decode_tps_list = [r["decode_tps"] for r in eval_results if r.get("decode_tps") is not None]
  all_chunk_itls = [itl for r in eval_results for itl in r["itls"]]

  total_prompt = sum(r["prompt_tokens"] for r in ok_results)
  total_cached = sum(r["cached_tokens"] for r in ok_results)
  total_uncached = max(0, total_prompt - total_cached)
  total_compl = sum(r["completion_tokens"] for r in ok_results)
  cache_hit_pct = (100.0 * total_cached / total_prompt) if total_prompt > 0 else 0.0
  tool_call_reqs = sum(1 for r in ok_results if r.get("has_tool_calls") or r.get("finish_reason") == "tool_calls")

  bucket_breakdown = {}
  for b_name, _, _, _ in ISL_BUCKETS:
    b_res = [r for r in ok_results if r["bucket"] == b_name]
    b_ttfts = [r["ttft"] for r in b_res if r["ttft"] is not None]
    b_tps = [r["decode_tps"] for r in b_res if r.get("decode_tps") is not None]
    b_p = sum(r["prompt_tokens"] for r in b_res)
    b_c = sum(r["cached_tokens"] for r in b_res)
    b_out = sum(r["completion_tokens"] for r in b_res)
    bucket_breakdown[b_name] = {
        "count": len(b_res),
        "mean_isl": (b_p / len(b_res)) if b_res else 0.0,
        "mean_osl": (b_out / len(b_res)) if b_res else 0.0,
        "cache_hit_pct": (100.0 * b_c / b_p) if b_p > 0 else 0.0,
        "median_ttft_s": statistics.median(b_ttfts) if b_ttfts else 0.0,
        "p90_ttft_s": percentile(b_ttfts, 90),
        "median_decode_tps": statistics.median(b_tps) if b_tps else 0.0,
    }

  for r in results:
    r.pop("chunk_times", None)

  summary = {
      "num_sessions": num_sessions,
      "turns_per_session": turns_per_session,
      "duration_per_n_s": duration_per_n,
      "total_requests": len(results),
      "successful_requests": len(ok_results),
      "tool_call_requests": tool_call_reqs,
      "window_completed_requests": window_completed_reqs,
      "failed_requests": len(err_results),
      "wall_clock_s": wall_clock,
      "steady_state_window_s": eff_window_s,
      "achieved_rps": len(ok_results) / wall_clock if wall_clock > 0 else 0.0,
      "tokens": {
          "mean_prompt_tokens": total_prompt / len(ok_results) if ok_results else 0.0,
          "mean_cached_tokens": total_cached / len(ok_results) if ok_results else 0.0,
          "mean_uncached_tokens": total_uncached / len(ok_results) if ok_results else 0.0,
          "mean_completion_tokens": total_compl / len(ok_results) if ok_results else 0.0,
          "observed_cache_hit_pct": cache_hit_pct,
      },
      "throughput": {
          "window_output_tok_per_s": window_output_tokens / eff_window_s if eff_window_s > 0 else 0.0,
          "window_input_tok_per_s": window_input_tokens / eff_window_s if eff_window_s > 0 else 0.0,
          "window_uncached_prefill_tok_per_s": window_uncached_tokens / eff_window_s if eff_window_s > 0 else 0.0,
          "output_tok_per_s": total_compl / wall_clock if wall_clock > 0 else 0.0,
          "input_tok_per_s": total_prompt / wall_clock if wall_clock > 0 else 0.0,
          "uncached_prefill_tok_per_s": total_uncached / wall_clock if wall_clock > 0 else 0.0,
          "total_tok_per_s": (total_prompt + total_compl) / wall_clock if wall_clock > 0 else 0.0,
      },
      "client_decode_tps_per_stream": {
          "mean": statistics.mean(decode_tps_list) if decode_tps_list else 0.0,
          "median": statistics.median(decode_tps_list) if decode_tps_list else 0.0,
          "p90": percentile(decode_tps_list, 90),
      },
      "per_token_itl_ms": {
          "mean": statistics.mean(per_tok_itls) * 1000.0 if per_tok_itls else 0.0,
          "median": statistics.median(per_tok_itls) * 1000.0 if per_tok_itls else 0.0,
          "p90": percentile(per_tok_itls, 90) * 1000.0,
      },
      "ttft_s": {
          "mean": statistics.mean(ttfts) if ttfts else 0.0,
          "median": statistics.median(ttfts) if ttfts else 0.0,
          "p90": percentile(ttfts, 90),
          "p99": percentile(ttfts, 99),
      },
      "chunk_itl_ms": {
          "mean": statistics.mean(all_chunk_itls) * 1000.0 if all_chunk_itls else 0.0,
          "median": statistics.median(all_chunk_itls) * 1000.0 if all_chunk_itls else 0.0,
          "p90": percentile(all_chunk_itls, 90) * 1000.0,
      },
      "e2e_s": {
          "mean": statistics.mean(e2es) if e2es else 0.0,
          "median": statistics.median(e2es) if e2es else 0.0,
          "p90": percentile(e2es, 90),
      },
      "buckets": bucket_breakdown,
  }

  mode_label = (
      f"Continuous {duration_per_n:.0f}s Window"
      if duration_per_n > 0
      else f"{turns_per_session} turn(s)/session"
  )
  print("=" * 94)
  print(
      f"RESULTS FOR n={num_sessions} SESSIONS ({mode_label}) | "
      f"OK={len(ok_results)}/{len(results)} (ToolCalls={tool_call_reqs}, In-Win={window_completed_reqs}) | "
      f"Win={eff_window_s:.1f}s | DrainWall={wall_clock:.2f}s"
  )
  print("-" * 94)
  print(
      f"  Mean ISL: {summary['tokens']['mean_prompt_tokens']:,.1f} tok | "
      f"Cached: {summary['tokens']['mean_cached_tokens']:,.1f} tok "
      f"({cache_hit_pct:.2f}%) | "
      f"Mean Output: {summary['tokens']['mean_completion_tokens']:,.1f} tok"
  )
  print(
      f"  Steady-State Window Tput : Output={summary['throughput']['window_output_tok_per_s']:,.1f} tok/s | "
      f"Input={summary['throughput']['window_input_tok_per_s']:,.1f} tok/s | "
      f"Uncached={summary['throughput']['window_uncached_prefill_tok_per_s']:,.1f} tok/s"
  )
  print(
      f"  Full-Drain Wall Tput     : Output={summary['throughput']['output_tok_per_s']:,.1f} tok/s | "
      f"Input={summary['throughput']['input_tok_per_s']:,.1f} tok/s | "
      f"RPS={summary['achieved_rps']:.2f}"
  )
  print(
      f"  Client Decode Speed/Strm : Median={summary['client_decode_tps_per_stream']['median']:.2f} tok/s "
      f"(Mean={summary['client_decode_tps_per_stream']['mean']:.2f}, P90={summary['client_decode_tps_per_stream']['p90']:.2f}) | "
      f"Per-Token ITL={summary['per_token_itl_ms']['median']:.2f}ms (StepChunk={summary['chunk_itl_ms']['median']:.2f}ms)"
  )
  print(
      f"  Latency                  : Median TTFT={summary['ttft_s']['median']:.2f}s (P90={summary['ttft_s']['p90']:.2f}s) | "
      f"Median E2E={summary['e2e_s']['median']:.2f}s"
  )
  print("  Per-Bucket Breakdown:")
  for b_name, b_stat in bucket_breakdown.items():
    print(
        f"    {b_name:8s}: reqs={b_stat['count']:3d} | Mean ISL={b_stat['mean_isl']:7,.0f} | "
        f"Mean OSL={b_stat['mean_osl']:6,.1f} | Hit={b_stat['cache_hit_pct']:5.1f}% | "
        f"Med TTFT={b_stat['median_ttft_s']:6.2f}s | P90 TTFT={b_stat['p90_ttft_s']:6.2f}s | "
        f"Med TPS/strm={b_stat['median_decode_tps']:5.1f}"
    )
  print("=" * 94)
  return summary


async def wait_for_health_and_set_state(
    http_session: aiohttp.ClientSession,
    prefill_url: str,
    decode_url: str,
    prefill_state_json: str,
    decode_state_json: str,
) -> None:
  for label, url in [("prefill", prefill_url), ("decode", decode_url)]:
    if not url:
      continue
    health_url = f"{url.rstrip('/')}/health"
    print(f"Waiting for {label} health at {health_url} ...")
    while True:
      try:
        async with http_session.get(health_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
          if resp.status == 200:
            print(f"  [{label}] Healthy (HTTP 200)!")
            break
      except Exception:
        pass
      await asyncio.sleep(5)

  for label, url, state_str in [
      ("prefill", prefill_url, prefill_state_json),
      ("decode", decode_url, decode_state_json),
  ]:
    if not url:
      continue
    try:
      async with http_session.post(
          f"{url.rstrip('/')}/abort_request",
          json={"rid": "", "abort_all": True},
          timeout=aiohttp.ClientTimeout(total=10),
      ) as _:
        pass
    except Exception:
      pass
    if not state_str:
      continue
    state_url = f"{url.rstrip('/')}/set_internal_state"
    payload = json.loads(state_str)
    print(f"Applying /set_internal_state to {label} ({state_url}): {json.dumps(payload)}")
    t0 = time.perf_counter()
    async with http_session.post(
        state_url, json=payload, timeout=aiohttp.ClientTimeout(total=1200)
    ) as resp:
      body = await resp.text()
      dt = time.perf_counter() - t0
      print(f"  [{label}] /set_internal_state returned HTTP {resp.status} in {dt:.2f}s: {body}")
      if resp.status != 200:
        raise RuntimeError(f"Failed /set_internal_state on {label}: HTTP {resp.status} {body}")
    try:
      async with http_session.post(
          f"{url.rstrip('/')}/abort_request",
          json={"rid": "", "abort_all": True},
          timeout=aiohttp.ClientTimeout(total=10),
      ) as _:
        pass
      await asyncio.sleep(4.0)
    except Exception:
      pass


async def _is_endpoint_healthy(http_session: aiohttp.ClientSession, url: str) -> bool:
  if not url:
    return True
  try:
    async with http_session.get(
        f"{url.rstrip('/')}/health", timeout=aiohttp.ClientTimeout(total=4)
    ) as resp:
      return resp.status == 200
  except Exception:
    return False


async def main_async(args: argparse.Namespace) -> None:
  session_list = [int(x.strip()) for x in args.sessions.split(",") if x.strip()]
  raw_reqs = load_raw_requests(args.dataset)
  cur_base_url = args.base_url
  cur_prefill_url = args.prefill_url
  cur_decode_url = args.decode_url
  switched_to_fallback = False

  timeout = aiohttp.ClientTimeout(total=args.timeout, sock_read=args.timeout)
  connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300, force_close=False)

  all_summaries = []
  async with aiohttp.ClientSession(timeout=timeout, connector=connector) as http_session:
    await wait_for_health_and_set_state(
        http_session,
        prefill_url=cur_prefill_url,
        decode_url=cur_decode_url,
        prefill_state_json=args.set_prefill_state_json,
        decode_state_json=args.set_decode_state_json,
    )
    for idx, n in enumerate(session_list):
      # Proactively cut over to the fresh duplicate server as soon as it finishes loading,
      # or if the primary server has reached its 2-hour pod timeout.
      if not switched_to_fallback and args.fallback_base_url:
        dup_ok = (
            await _is_endpoint_healthy(http_session, args.fallback_base_url)
            and await _is_endpoint_healthy(http_session, args.fallback_prefill_url)
            and await _is_endpoint_healthy(http_session, args.fallback_decode_url)
        )
        prim_ok = (
            await _is_endpoint_healthy(http_session, cur_base_url)
            and await _is_endpoint_healthy(http_session, cur_prefill_url)
            and await _is_endpoint_healthy(http_session, cur_decode_url)
        )
        if dup_ok or not prim_ok:
          if not dup_ok:
            print(f"[FAILOVER] Primary server unreachable before n={n}; waiting for duplicate server {args.fallback_base_url} to finish loading...")
            await wait_for_health_and_set_state(
                http_session,
                prefill_url=args.fallback_prefill_url or args.fallback_base_url,
                decode_url=args.fallback_decode_url,
                prefill_state_json="",
                decode_state_json="",
            )
          print(f"\n[FAILOVER] Switching benchmark traffic to fresh duplicate server: {args.fallback_base_url}")
          cur_base_url = args.fallback_base_url
          cur_prefill_url = args.fallback_prefill_url or args.fallback_base_url
          cur_decode_url = args.fallback_decode_url
          switched_to_fallback = True

      target_url = f"{cur_base_url.rstrip('/')}/v1/chat/completions"
      print(f"\n>>> [n={n}] 3.5-Min ({args.duration_per_n:.0f}s) Measured Benchmark Run (1-8K & 32-64K OSL=1024, Zero Compilation) <<<")
      s = await run_single_n(
          http_session,
          base_url=cur_base_url,
          prefill_url=cur_prefill_url,
          decode_url=cur_decode_url,
          target_url=target_url,
          raw_reqs=raw_reqs,
          num_sessions=n,
          turns_per_session=args.turns_per_session,
          duration_per_n=args.duration_per_n,
          cache_hit_ratio=args.cache_hit_rate,
          max_tokens=args.max_tokens,
          ignore_eos=args.ignore_eos,
          dp_size=args.dp_size,
          flush_before_warmup=((idx == 0 and not args.no_flush_on_start) or args.flush_each_n),
          warmup_concurrency=args.warmup_concurrency,
      )
      s["pass"] = "measured_210s" 
      if s.get("failed_requests", 0) > 0 and not switched_to_fallback and args.fallback_base_url:
        print(f"\n[FAILOVER] Detected {s.get('failed_requests')} failed requests during n={n} (primary 2h timeout). Failing over to {args.fallback_base_url} and re-running n={n}...")
        await wait_for_health_and_set_state(
            http_session,
            prefill_url=args.fallback_prefill_url or args.fallback_base_url,
            decode_url=args.fallback_decode_url,
            prefill_state_json="",
            decode_state_json="",
        )
        cur_base_url = args.fallback_base_url
        cur_prefill_url = args.fallback_prefill_url or args.fallback_base_url
        cur_decode_url = args.fallback_decode_url
        switched_to_fallback = True
        target_url = f"{cur_base_url.rstrip('/')}/v1/chat/completions"
        s = await run_single_n(
            http_session,
            base_url=cur_base_url,
            prefill_url=cur_prefill_url,
            decode_url=cur_decode_url,
            target_url=target_url,
            raw_reqs=raw_reqs,
            num_sessions=n,
            turns_per_session=args.turns_per_session,
            duration_per_n=args.duration_per_n,
            cache_hit_ratio=args.cache_hit_rate,
            max_tokens=args.max_tokens,
            ignore_eos=args.ignore_eos,
            dp_size=args.dp_size,
            flush_before_warmup=True,
            warmup_concurrency=args.warmup_concurrency,
        )
      all_summaries.append(s)

  if args.output_json:
    out_p = Path(args.output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
      json.dump(all_summaries, f, indent=2)
    print(f"\nSaved full sweep summary to {out_p}")


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="N-Session Empirical ISL + 80% Cache Hit Benchmark")
  p.add_argument("--base-url", type=str, default="http://glm52-pdb-lb-svc.vllm-serving.svc.cluster.local:8000")
  p.add_argument("--prefill-url", type=str, default="http://glm52-pdb-prefill-svc.vllm-serving.svc.cluster.local:30271")
  p.add_argument("--decode-url", type=str, default="http://glm52-pdb-decode-svc.vllm-serving.svc.cluster.local:30271")
  p.add_argument("--fallback-base-url", type=str, default="")
  p.add_argument("--fallback-prefill-url", type=str, default="")
  p.add_argument("--fallback-decode-url", type=str, default="")
  p.add_argument("--set-prefill-state-json", type=str, default="")
  p.add_argument("--set-decode-state-json", type=str, default="")
  p.add_argument("--dataset", type=str, default="/data/requests_mocked.jsonl.xz")
  p.add_argument("--sessions", type=str, default="1,2,4,8,16,32,64,128")
  p.add_argument("--turns-per-session", type=int, default=1)
  p.add_argument(
      "--duration-per-n",
      type=float,
      default=0.0,
      help="If > 0, run each concurrency n continuously in a closed loop for this many seconds.",
  )
  p.add_argument("--cache-hit-rate", type=float, default=0.80)
  p.add_argument(
      "--max-tokens",
      type=int,
      default=0,
      help="If > 0, override max_tokens; if 0 (default), use each request's max_tokens from requests_mocked.jsonl.",
  )
  p.add_argument("--dp-size", type=int, default=1)
  p.add_argument("--ignore-eos", action="store_true", default=False)
  p.add_argument("--no-ignore-eos", action="store_false", dest="ignore_eos")
  p.add_argument("--flush-each-n", action="store_true", default=True)
  p.add_argument("--no-flush-each-n", action="store_false", dest="flush_each_n")
  p.add_argument("--no-flush-on-start", action="store_true", default=False)
  p.add_argument("--warmup-on-start", action="store_true", default=False)
  p.add_argument("--warmup-concurrency", type=int, default=16)
  p.add_argument("--timeout", type=float, default=3600.0)
  p.add_argument("--output-json", type=str, default="/tmp/pdb_sessions_sweep.json")
  return p.parse_args()


if __name__ == "__main__":
  asyncio.run(main_async(parse_args()))
