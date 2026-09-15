"""Explicit, per-dataset LLM grading after target-model generation is saved.

The transport accepts a chat-completions HTTP endpoint. Dataset functions own
the messages and score interpretation; the framework owns persistence, retry,
metric validation and harness aggregation. No judge is selected automatically.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from http.client import HTTPException
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def validate_judge_config(task: dict, protocol: dict) -> None:
    """Reject ambiguous modes and invalid transport settings before model loading."""
    mode = protocol.get('scoring', 'standard')
    if mode not in ('standard', 'llm_judge'):
        raise ValueError('scoring must be standard or llm_judge')
    if mode == 'standard':
        if 'judge' in protocol:
            raise ValueError('judge settings require scoring: llm_judge')
        return
    if task.get('output_type') != 'generate_until':
        raise ValueError('llm_judge requires generate_until (grade generated answers)')
    if 'process_results' in task:
        raise ValueError('llm_judge uses judge.score; remove top-level process_results')
    judge = protocol.get('judge')
    if not isinstance(judge, dict):
        raise ValueError('llm_judge requires a judge mapping')
    allowed = {'endpoint', 'model', 'api_key_env', 'prompt', 'score',
               'generation_kwargs', 'timeout', 'max_retries'}
    if set(judge) - allowed:
        raise ValueError(f'unknown judge settings: {sorted(set(judge) - allowed)}')
    for key in ('endpoint', 'model', 'prompt', 'score'):
        if not isinstance(judge.get(key), str) or not judge[key].strip():
            raise ValueError(f'judge.{key} is required')
    url = urlsplit(judge['endpoint'])
    if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError('judge.endpoint must be an HTTP(S) URL without credentials/query/fragment')
    key_env = judge.get('api_key_env')
    if key_env is not None and (not isinstance(key_env, str) or not key_env.isidentifier()):
        raise ValueError('judge.api_key_env must name an environment variable')
    timeout = judge.get('timeout', 60)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('judge.timeout must be positive and finite')
    retries = judge.get('max_retries', 2)
    if type(retries) is not int or not 0 <= retries <= 10:
        raise ValueError('judge.max_retries must be an integer between 0 and 10')
    kwargs = judge.get('generation_kwargs', {})
    if not isinstance(kwargs, dict) or {'model', 'messages', 'stream'}.intersection(kwargs):
        raise ValueError('judge.generation_kwargs cannot override model/messages/stream')
    json.dumps(kwargs, allow_nan=False)
    metrics = task.get('metric_list', [])
    if any(not isinstance(m.get('metric'), str) for m in metrics):
        raise ValueError('judge metrics must have string names')


def pending_judge_results(doc, results):
    """Return no provisional scores: ungraded answers must never appear as zero."""
    return {}


def _write_json(path: Path, value: dict) -> None:
    """Replace a checkpoint atomically, keeping a complete file after interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def _reject_nonfinite(token: str):
    raise ValueError(f'judge HTTP response contains non-finite JSON number {token}')


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        _reject_nonfinite(text)
    return value


def _safe_http_error_detail(error: HTTPError, payload: dict) -> str:
    """Expose only status and a request-key name, never the server's message."""
    detail = f"HTTP {error.code}"
    try:
        body = json.load(error, parse_constant=_reject_nonfinite, parse_float=_finite_float)
        parameter = body.get('error', {}).get('param') if isinstance(body, dict) else None
    except Exception:
        parameter = None
    # The response may be hostile or echo secrets. A key already present in our
    # payload is safe to name; arbitrary response strings are not.
    if isinstance(parameter, str) and parameter in payload:
        detail += f", parameter {parameter!r}"
    return detail


def request_judge(config: dict, messages: list) -> dict:
    """Call an explicitly configured endpoint; only transient transport errors retry.

    API keys are read at request time and never included in saved configuration.
    Invalid model output is preserved for diagnosis instead of sampling repeatedly
    until a parseable (and potentially biased) grade appears.
    """
    headers = {'Content-Type': 'application/json'}
    if config.get('api_key_env'):
        key = os.environ.get(config['api_key_env'])
        if not key:
            raise ValueError(f"missing judge API key environment variable: {config['api_key_env']}")
        headers['Authorization'] = f'Bearer {key}'
    payload = {'model': config['model'], 'messages': messages, 'stream': False,
               **config.get('generation_kwargs', {})}
    request = Request(config['endpoint'], data=json.dumps(payload, allow_nan=False).encode(),
                      headers=headers, method='POST')
    retries = config.get('max_retries', 2)
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=config.get('timeout', 60)) as response:
                # NaN/Infinity would later make the failure record unwritable.
                result = json.load(response, parse_constant=_reject_nonfinite,
                                   parse_float=_finite_float)
            if not isinstance(result, dict):
                raise ValueError('judge HTTP response must be a JSON object')
            return result
        # urllib raises a dropped connection or truncated body outside URLError.
        except (HTTPError, URLError, TimeoutError, ConnectionError, HTTPException) as error:
            transient = not isinstance(error, HTTPError) or error.code in (408, 429, 500, 502, 503, 504)
            if not transient or attempt == retries:
                # Do not expose server messages, which could echo credentials.
                detail = _safe_http_error_detail(error, payload) if isinstance(error, HTTPError) else None
                suffix = f" ({detail})" if detail else ""
                raise RuntimeError(f'judge request failed{suffix}; no score was assigned') from None
            time.sleep(min(2 ** attempt, 8))
    raise AssertionError('unreachable')


def _validate_scores(scores, metric_names: set, correctness: str | None) -> None:
    """Require the declared finite scalar metrics, with explicit binary correctness."""
    if not isinstance(scores, dict) or set(scores) != metric_names:
        raise ValueError('judge.score must return exactly the metric_list names')
    for name, value in scores.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'judge metric {name} must be a finite number')
    if correctness and scores[correctness] not in (0, 1):
        raise ValueError('judge correctness_metric must be binary 0/1')


def grade_sample(sample: dict, protocol: dict, metric_names: set,
                 run_dir: Path, provenance: dict) -> dict:
    """Grade one filtered answer, reusing only a successful identical judge request.

    Full document, filtered answers, rendered messages, protocol and source hashes
    participate in cache identity. A changed rubric, parser or reference answer
    therefore cannot silently reuse an old score. The response is saved before
    calling user score code, so parsing failures remain inspectable.
    """
    judge = protocol['judge']
    messages = judge['prompt'](sample['doc'], sample['filtered_resps'])
    if not isinstance(messages, list) or not messages or any(
        not isinstance(m, dict) or m.get('role') not in ('system', 'user', 'assistant')
        or not isinstance(m.get('content'), str) for m in messages
    ):
        raise ValueError('judge.prompt must return nonempty chat messages with role/content')
    transport = {key: value for key, value in judge.items() if key not in ('prompt', 'score')}
    identity = {'provenance': provenance, 'judge': transport, 'task': sample.get('task_name'),
                'doc': sample['doc'], 'responses': sample['filtered_resps'],
                'filter': sample.get('filter', 'none'), 'messages': messages,
                'metrics': sorted(metric_names)}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False,
                                        allow_nan=False).encode()).hexdigest()
    path = run_dir / 'judge' / f'{digest}.json'
    if path.exists():
        record = json.loads(path.read_text())
        if record.get('status') == 'success':
            _validate_scores(record['scores'], metric_names, protocol.get('correctness_metric'))
            sample['_judge'] = {'status': 'success', 'record': str(path.relative_to(run_dir)),
                                'cache_hit': True}
            return record['scores']
    # Keep the service call time outside the cache identity: rerunning an identical
    # request may reuse this record, while the timestamp documents the original call.
    record = {'status': 'pending', 'identity': identity,
              'requested_at': datetime.now(timezone.utc).isoformat()}
    _write_json(path, record)
    try:
        record['response'] = request_judge(judge, messages)
        _write_json(path, record)
        scores = judge['score'](sample['doc'], sample['filtered_resps'], record['response'])
        _validate_scores(scores, metric_names, protocol.get('correctness_metric'))
        record.update(status='success', scores=scores)
    except Exception as error:
        record.update(status='failed', error_type=type(error).__name__)
        _write_json(path, record)
        sample['_judge'] = {'status': 'failed', 'record': str(path.relative_to(run_dir))}
        raise
    _write_json(path, record)
    sample['_judge'] = {'status': 'success', 'record': str(path.relative_to(run_dir)),
                        'cache_hit': False}
    return scores


def score_judge_tasks(results: dict, tasks: list, run_dir, provenance: dict) -> None:
    """Persist all generations first, then grade and aggregate through lm-eval.

    On failure the run stays incomplete; successful per-sample scores remain in
    the journal and no partial aggregate is published. Ordinary tasks are untouched.
    """
    from . import storage

    judges = [task for task in tasks if not isinstance(task, (str, dict))
              and (task.config.metadata or {}).get('eval_framework', {}).get('scoring') == 'llm_judge']
    if not judges:
        return
    samples_by_task = results.get('samples', {})
    for task in judges:
        protocol = task.config.metadata['eval_framework']
        for sample in samples_by_task.get(task.config.task, []):
            sample['_judge'] = {'status': 'pending'}
            sample['_eval_framework'] = {
                'sample_id': sample['doc'][protocol['sample_id']],
                'primary_filter': protocol['primary_filter'], 'is_correct': None,
            }
            for metric in task.aggregation():
                sample.pop(metric, None)
            sample['metrics'] = []
    storage.write_samples(run_dir, samples_by_task)
    try:
        for task in judges:
            name = task.config.task
            protocol = task.config.metadata['eval_framework']
            sample_metrics = defaultdict(list)
            for sample in samples_by_task.get(name, []):
                grading_sample = {**sample, 'task_name': name}
                try:
                    scores = grade_sample(grading_sample, protocol, set(task.aggregation()),
                                          Path(run_dir), provenance)
                except Exception:
                    grading_sample['_judge']['status'] = 'failed'
                    raise
                finally:
                    sample['_judge'] = grading_sample['_judge']
                sample.update(scores)
                sample['metrics'] = list(scores)
                if protocol.get('correctness_metric'):
                    sample['_eval_framework']['is_correct'] = bool(scores[protocol['correctness_metric']])
                for metric, value in scores.items():
                    sample_metrics[(metric, sample.get('filter', 'none'))].append(value)
            # lm-eval 0.4.13 extracted this helper from the 0.4.9.1 TaskOutput.
            # Use each installed version's own aggregation and stderr behavior.
            try:
                from lm_eval.evaluator_utils import _compute_task_aggregations
            except ImportError:
                from lm_eval.evaluator_utils import TaskOutput
                output = TaskOutput.from_taskdict(name, task)
                output.sample_metrics = sample_metrics
                output.calculate_aggregate_metric()
                aggregate, count = output.agg_metrics, output.sample_len
                count_key = 'samples'
            else:
                aggregate, count = _compute_task_aggregations(task, sample_metrics, 100000)
                count_key = 'sample_len'
            results['results'].setdefault(name, {}).update({**aggregate, count_key: count})
            if name in results.get('n-samples', {}):
                results['n-samples'][name]['effective'] = count
    finally:
        storage.write_samples(run_dir, samples_by_task)
