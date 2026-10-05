"""Typed judgments from a decision model (TypeSafe "System One", model Jev): client, cache,
counters, fallback, and an offline surrogate. Questions and thresholds live in bmira/questions.py.

The judge runs in SHADOW mode only: every touchpoint (bmira/shadow.py) asks, logs the answer next
to what today's executor decided, and changes nothing. A touchpoint acts only after its thresholds
are calibrated on gold labels (handoff 4.5); `judge_mode="act"` is refused until then.

Shadow mode must never cost a run: every failure is caught per item (`ask_many` returns None for
it), and an authentication error switches the judge off for the rest of the run.

API shape (TypeSafe docs, checked 2026-10-05, not yet confirmed against a live response):
POST {state, model, questions} -> {"answers": {qid: answer}, "usage": {"input_tokens": n}}.
Answers: noul -> {"noul": p_yes}; choice -> {"choice", "probabilities", "confidence"};
score -> {"score", "probabilities", "legend", "confidence"}. Pin a recorded response as a test
fixture on first live use.
"""
import hashlib
import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from bmira import questions as Q

API = "https://api.typesafe.ai/v1/systemone"
RETRYABLE = {429, 500, 502, 503, 529}
FATAL = {401, 403, 404}          # bad key, no access, unknown model: retrying cannot help
MOVING_ALIASES = ("-latest", "-preview")


class JudgeUnavailable(Exception):
    pass


class _Counted:
    """Per-task counters, read by bmira.telemetry beside the LLM's."""

    def _init_counters(self, model):
        self.calls, self.items, self.failures = Counter(), Counter(), Counter()
        self.tokens_in, self.seconds, self.cache_hits = Counter(), Counter(), Counter()
        self.model_of, self.model, self.disabled = {}, model, ""
        self._lock = threading.Lock()

    def ask_many(self, task: str, items: list) -> list:
        """[(state, questions[, ctx])] -> [answers or None], in parallel; one failure costs one item."""
        def one(x):
            try:
                return self.ask(task, *x)
            except JudgeUnavailable:
                return None
        if not items:
            return []
        with ThreadPoolExecutor(max(1, min(self.workers, len(items)))) as ex:
            return list(ex.map(one, items))

    def save(self):
        pass


class JevJudge(_Counted):
    def __init__(self, settings, api_key: str | None = None):
        self.s = settings
        self.key = api_key or os.environ.get("TYPESAFE_API_KEY", "")
        if not self.key:
            raise RuntimeError("TYPESAFE_API_KEY missing; set judge_provider='off' to run without the judge")
        self._init_counters(settings.judge_model)
        self.workers = settings.judge_workers
        self.path = Path(settings.cache_dir) / f"judge_{settings.judge_model}.json" if settings.cache_dir else None
        self.cache = self._load()
        self._dirty = False

    def _load(self) -> dict:
        if not (self.path and self.path.exists()):
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            print(f"[judge][WARN] cache {self.path} unreadable; starting empty")
            return {}

    def _key(self, task, state, questions) -> str:
        blob = json.dumps([self.s.judge_model, task, state, questions], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def ask(self, task: str, state, questions: dict, ctx=None) -> dict:
        """One request: every question against one state. {qid: answer}; raises JudgeUnavailable."""
        if self.disabled:
            raise JudgeUnavailable(f"{task}: judge disabled ({self.disabled})")
        k = self._key(task, state, questions)
        with self._lock:
            if k in self.cache:
                self.cache_hits[task] += 1
                return self.cache[k]
            self.calls[task] += 1
            self.items[task] += len(questions)
            self.model_of[task] = self.s.judge_model
        t0 = time.perf_counter()
        try:
            for attempt in range(self.s.judge_max_retries):
                try:
                    r = requests.post(API, timeout=self.s.judge_timeout_s,
                                      headers={"Authorization": f"Bearer {self.key}"},
                                      json={"state": state, "model": self.s.judge_model, "questions": questions})
                except (requests.ConnectionError, requests.Timeout):
                    time.sleep(min(2 ** attempt, 30))
                    continue
                if r.status_code in RETRYABLE:
                    time.sleep(min(2 ** attempt, 30))
                    continue
                if r.status_code in FATAL:
                    with self._lock:
                        self.disabled = f"HTTP {r.status_code}"
                    print(f"[judge][WARN] HTTP {r.status_code}: judge switched off for this run")
                    raise JudgeUnavailable(f"{task}: HTTP {r.status_code}")
                r.raise_for_status()
                body = r.json()
                answers = body["answers"]
                if not isinstance(answers, dict):
                    raise ValueError("answers is not an object")
                with self._lock:
                    self.tokens_in[task] += (body.get("usage") or {}).get("input_tokens", 0) or 0
                    self.cache[k] = answers
                    self._dirty = True
                return answers
            raise JudgeUnavailable(f"{task}: retries exhausted")
        except JudgeUnavailable:
            with self._lock:
                self.failures[task] += 1
            raise
        except (requests.RequestException, KeyError, ValueError) as e:
            with self._lock:
                self.failures[task] += 1
            raise JudgeUnavailable(f"{task}: {type(e).__name__}") from e
        finally:
            with self._lock:
                self.seconds[task] += time.perf_counter() - t0

    def save(self):
        """Write the cache atomically, so replays at the same model version are free and identical."""
        if not (self.path and self._dirty):
            return
        with self._lock:
            blob = json.dumps(self.cache, ensure_ascii=False)
            self._dirty = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(blob, encoding="utf-8")
        tmp.replace(self.path)


class SurrogateJudge(_Counted):
    """Offline stand-in. Reads `ctx`, never the state: ctx["current"] maps a question id to what today's
    executor decided, and the surrogate agrees with it at probability `p` (0.5 where nothing is known),
    so offline runs and tests are deterministic. `disagree` lists question ids it answers the other way."""
    workers = 4

    def __init__(self, p: float = 0.9, disagree=()):
        self._init_counters("surrogate")
        self.p, self.disagree = p, set(disagree)

    def ask(self, task, state, questions, ctx=None):
        with self._lock:
            self.calls[task] += 1
            self.items[task] += len(questions)
            self.model_of[task] = "surrogate"
            self.tokens_in[task] += len(json.dumps(state, default=str)) // 4
        current = (ctx or {}).get("current", {})
        return {qid: self._answer(q, current.get(qid), qid in self.disagree) for qid, q in questions.items()}

    def _answer(self, q, cur, flip):
        p = self.p if cur is not None else 0.5
        if q["type"] == "noul":
            yes = bool(cur) != flip if cur is not None else True
            return {"noul": p if yes else 1 - p}
        opts = list(q["criteria"]) if isinstance(q["criteria"], dict) else list(range(len(q["criteria"])))
        idx = opts.index(cur) if cur in opts else 0
        if flip:
            idx = (idx + 1) % len(opts)
        rest = (1 - p) / max(1, len(opts) - 1)
        probs = [p if i == idx else rest for i in range(len(opts))]
        if q["type"] == "choice":
            return {"choice": opts[idx], "probabilities": dict(zip(opts, probs)),
                    "confidence": round((p - 1 / len(opts)) / (1 - 1 / len(opts)), 3)}
        return {"score": idx, "probabilities": probs, "legend": q["criteria"], "confidence": p}


def make_judge(settings):
    """The judge for these settings: None when off. Live runs only; offline runs use SurrogateJudge."""
    if settings.judge_provider == "off":
        return None
    if settings.judge_mode != "shadow":
        raise ValueError("judge_mode must be 'shadow': a touchpoint acts only after calibration (handoff 4.5)")
    if settings.judge_provider != "jev":
        raise ValueError(f"unknown judge_provider {settings.judge_provider!r}; use 'jev' or 'off'")
    if settings.judge_model.endswith(MOVING_ALIASES):
        raise ValueError(f"judge_model {settings.judge_model!r} is a moving alias; pin a version")
    if settings.judge_model != Q.MODEL:
        print(f"[judge][WARN] questions.py is pinned to {Q.MODEL}, not {settings.judge_model}: "
              "thresholds are uncalibrated for this model")
    return JevJudge(settings)


# ── reading answers ─────────────────────────────────────────────────────────
def p_yes(a: dict | None) -> float | None:
    v = None if a is None else a.get("noul")
    return float(v) if isinstance(v, (int, float)) else None


def top(a: dict | None):
    """(choice, confidence, probabilities) of a Choice answer; (None, 0.0, {}) if missing."""
    if not a or a.get("choice") is None:
        return None, 0.0, {}
    return a["choice"], float(a.get("confidence") or 0.0), dict(a.get("probabilities") or {})


def level_probs(a: dict | None, criteria: list) -> list[float] | None:
    """P(level i), i = 0 .. n-1, from a Score answer. Accepts probabilities as a list, as a dict keyed by
    level number (0- or 1-based) or by criterion text; falls back to a point mass at the reported score."""
    n = len(criteria)
    if not a:
        return None
    probs = a.get("probabilities")
    if isinstance(probs, list) and len(probs) == n:
        out = [float(x) for x in probs]
    elif isinstance(probs, dict) and probs:
        if all(k in probs for k in criteria):
            out = [float(probs[k]) for k in criteria]
        else:
            try:
                nums = {int(k): float(v) for k, v in probs.items()}
            except (TypeError, ValueError):
                return None
            base = 1 if min(nums) == 1 and max(nums) == n else 0
            out = [nums.get(i + base, 0.0) for i in range(n)]
    elif isinstance(a.get("score"), (int, float)):
        level = round(a["score"])
        level = level - 1 if level == n else level           # a 1-based score at the top level
        out = [1.0 if i == max(0, min(n - 1, level)) else 0.0 for i in range(n)]
    else:
        return None
    total = sum(out)
    return [x / total for x in out] if total > 0 else None


def expected_level(probs: list[float] | None) -> float | None:
    return None if probs is None else sum(i * p for i, p in enumerate(probs))
