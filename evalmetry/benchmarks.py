"""Local JSONL benchmark contracts layered on lm-eval task configurations.

A bundle contains YAML tasks, local Python scoring code, and JSONL data. Paths
are relative to each YAML, never the shell's working directory. User functions
are executable Python and must come from a trusted bundle.
"""
from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

import yaml


class _ConfigLoader(yaml.SafeLoader):
    """Inspect function references without executing user code."""


_ConfigLoader.add_constructor("!function", lambda loader, node: loader.construct_scalar(node))


def prepare_benchmarks(
    include_paths: list[str], tasks: list[str],
) -> tuple[Any, list[Any], dict[str, Any]]:
    """Validate local tasks before model loading and return manager/configs/provenance.

    The first contract supports explicit task names and local JSONL splits only.
    ``metadata.eval_framework`` declares sample_id, primary_metric,
    primary_filter, higher_is_better, and optional correctness_metric. All Python
    and YAML in each include directory and all declared data/source_files are
    content-hashed. External helper code must be listed in source_files.
    """
    from lm_eval.tasks import TaskManager

    if not include_paths:
        return None, tasks, {}
    roots = [Path(p).expanduser().resolve() for p in include_paths]
    builtin = set(TaskManager().all_tasks)
    configs, files = _discover_local_tasks(roots, builtin)
    selected, protocols = [], {}
    for name in tasks:
        if name not in configs:
            if name not in builtin:
                raise ValueError(f"unknown task: {name}; use explicit task names")
            selected.append(name)
            continue

        path, raw = configs[name]
        protocol = _validate_task_protocol(name, raw)
        resolved, data_files = _resolve_split_files(name, path, raw, protocol)
        files.update(data_files)
        for source in protocol.get('source_files', []):
            files.add((path.parent / source).resolve())
        selected.append(_load_local_task(name, path, resolved, protocol))
        protocols[name] = protocol

    provenance = _build_provenance(files, protocols)
    manager = TaskManager(include_path=[str(p) for p in roots])
    return manager, selected, provenance


def _discover_local_tasks(
    roots: list[Path], builtin: set[str],
) -> tuple[dict[str, tuple[Path, dict[str, Any]]], set[Path]]:
    """모든 bundle의 task 이름과 hash 대상 코드를 찾는다. !function은 실행하지 않는다.

    선택하지 않은 YAML도 이름 충돌을 검사하고, 모든 Python/YAML을 hash 대상에
    넣는다. 반면 JSONL과 source_files는 선택된 task에서 참조한 파일만 추가한다.
    """
    configs: dict[str, tuple[Path, dict[str, Any]]] = {}
    files: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"include-path is not a directory: {root}")
        files.update(p for p in root.rglob('*') if p.suffix in {'.py', '.yaml', '.yml'})
        for path in sorted(list(root.rglob('*.yaml')) + list(root.rglob('*.yml'))):
            raw = yaml.load(path.read_text(), Loader=_ConfigLoader)
            if not isinstance(raw, dict) or not isinstance(raw.get('task'), str):
                raise ValueError(f"{path}: expected a named task YAML (groups/includes unsupported)")
            name = raw['task']
            if name in configs or name in builtin:
                raise ValueError(f"duplicate task name: {name} ({path})")
            configs[name] = (path, raw)
    return configs, files


def _validate_task_protocol(name: str, raw: dict[str, Any]) -> dict[str, Any]:
    """task 종류와 채점·filter 계약을 검증한다. 파일 읽기와 사용자 함수 로딩은 뒤에 한다.

    여러 설정이 잘못된 경우에도 기존과 같은 오류가 먼저 나오도록 검사 순서를
    유지한다. higher_is_better는 0/1 숫자가 아닌 bool이어야 한다.
    """
    if 'include' in raw or raw.get('dataset_path') != 'json':
        raise ValueError(f"{name}: use a self-contained local JSONL task")
    kind = raw.get('output_type')
    if kind not in ('multiple_choice', 'generate_until'):
        raise ValueError(f"{name}: unsupported output_type {kind}")
    protocol = dict(raw.get('metadata', {}).get('eval_framework', {}))
    from .judges import validate_judge_config
    validate_judge_config(raw, protocol)
    for field in ('sample_id', 'primary_metric', 'primary_filter', 'higher_is_better'):
        if field not in protocol:
            raise ValueError(f"{name}: metadata.eval_framework.{field} is required")
    metrics = {m['metric']: m for m in raw.get('metric_list', [])}
    primary = protocol['primary_metric']
    correctness = protocol.get('correctness_metric')
    if primary not in metrics or (correctness and correctness not in metrics):
        raise ValueError(f"{name}: declared metric is absent from metric_list")
    direction = protocol['higher_is_better']
    if type(direction) is not bool or metrics[primary].get('higher_is_better') != direction:
        raise ValueError(f"{name}: primary metric direction must match metric_list")
    filters = [f['name'] for f in raw.get('filter_list', [{'name': 'none'}])]
    if protocol['primary_filter'] not in filters:
        raise ValueError(f"{name}: unknown primary_filter")
    return protocol


def _resolve_split_files(
    name: str, path: Path, raw: dict[str, Any], protocol: dict[str, Any],
) -> tuple[dict[str, list[str]], set[Path]]:
    """YAML 기준으로 데이터 경로를 풀고 각 split의 모든 JSONL 문서를 검증한다.

    seen은 split마다 새로 만든다. 한 split이 여러 파일로 나뉘어도 ID 중복을
    잡고, 서로 다른 split에서 같은 ID를 쓰는 것은 허용한다. 빈 줄은 건너뛰되
    오류 위치는 실제 파일 줄 번호로 남긴다.
    """
    kind = raw['output_type']
    files: set[Path] = set()
    splits = raw.get('dataset_kwargs', {}).get('data_files', {})
    evaluation_split = raw.get('test_split', raw.get('validation_split'))
    if not isinstance(splits, dict) or not splits or evaluation_split not in splits:
        raise ValueError(f"{name}: data_files must declare the evaluation split")
    resolved = {}
    for split, paths in splits.items():
        paths = [paths] if isinstance(paths, str) else paths
        seen = set()
        resolved[split] = []
        for item in paths:
            data_path = (path.parent / item).resolve()
            files.add(data_path)
            resolved[split].append(str(data_path))
            with data_path.open() as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    doc = json.loads(line)
                    validate_document(doc, kind, protocol['sample_id'], seen,
                                      f"{data_path}:{line_number}",
                                      answer_optional=protocol.get('scoring') == 'llm_judge')
        if not seen:
            raise ValueError(f"{name}: empty split {split}")
    return resolved, files


def _load_local_task(
    name: str, path: Path, resolved: dict[str, list[str]], protocol: dict[str, Any],
) -> dict[str, Any]:
    """검증된 task를 lm-eval loader로 읽는다. 이 단계에서 !function 코드가 로딩된다.

    judge task는 평가 중에 채점하지 않도록 process_results를 바꾼다.
    실제 judge 요청은 생성 결과를 확보한 뒤 main.py의 평가 흐름에서 수행한다.
    """
    # Use the installed harness loader, including its native !function support.
    try:
        from lm_eval.tasks._yaml_loader import load_yaml
        loaded = load_yaml(path)
    except ImportError:
        from lm_eval.utils import load_yaml_config
        loaded = load_yaml_config(str(path))
    loaded['dataset_kwargs']['data_files'] = resolved
    if protocol.get('scoring') == 'llm_judge':
        from .judges import pending_judge_results
        judge = loaded['metadata']['eval_framework']['judge']
        if not callable(judge['prompt']) or not callable(judge['score']):
            raise ValueError(f"{name}: judge.prompt and judge.score must use !function")
        # Generation finishes and is saved before any judge request is made.
        loaded['process_results'] = pending_judge_results
        loaded.setdefault('doc_to_target', "{{ answer if answer is defined and answer is not none else '' }}")
    return loaded


def _build_provenance(files: set[Path], protocols: dict[str, Any]) -> dict[str, Any]:
    """선택된 계약, 관련 파일 내용, 의존성 버전을 재현성 정보로 기록한다.

    경로와 패키지 이름을 정렬해 해당 항목의 기록 순서를 고정한다. 큰 데이터
    파일도 통째로 읽지 않고 chunk 단위로 hash에 반영한다.
    """
    hashes = {}
    for path in sorted(files):
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        hashes[str(path)] = digest.hexdigest()
    dependencies = {"datasets", "jinja2", "PyYAML"}
    for protocol in protocols.values():
        dependencies.update(protocol.get("dependencies", []))
    provenance = {'files': hashes, 'protocols': protocols,
                  'python': platform.python_version(),
                  'packages': {name: version(name) for name in sorted(dependencies)}}
    return provenance


def validate_document(doc: dict, kind: str, id_field: str, seen: set, location: str,
                      *, answer_optional: bool = False) -> None:
    """Check template fields and unique stable IDs across one complete split.

    `seen` is shared across files of the same split and updated in place. IDs are
    compared as strings, so integer 1 and string "1" refer to the same sample.
    `answer_optional` permits an absent judge reference, not a malformed one.
    """
    if not isinstance(doc, dict):
        raise ValueError(f"{location}: expected a JSON object")
    sample_id = doc.get(id_field)
    if type(sample_id) not in (str, int) or sample_id == '':
        raise ValueError(f"{location}: {id_field} must be a nonempty string or integer")
    key = str(sample_id)
    if key in seen:
        raise ValueError(f"{location}: duplicate sample ID {key}")
    seen.add(key)
    text_field = 'question' if kind == 'multiple_choice' else 'prompt'
    if not isinstance(doc.get(text_field), str) or not doc[text_field].strip():
        raise ValueError(f"{location}: missing/nonempty {text_field} required")
    if kind == 'multiple_choice':
        choices, label = doc.get('choices'), doc.get('label')
        if (not isinstance(choices, list) or len(choices) < 2
                or any(not isinstance(choice, str) or not choice.strip() for choice in choices)):
            raise ValueError(f"{location}: choices must contain at least two nonempty strings")
        if type(label) is not int or not 0 <= label < len(choices):
            raise ValueError(f"{location}: label is outside choices")
    else:
        if answer_optional and 'answer' not in doc:
            return
        answer = doc.get('answer')
        if (not isinstance(answer, (str, list))
                or (isinstance(answer, list)
                    and (not answer or any(not isinstance(item, str) for item in answer)))):
            raise ValueError(f"{location}: answer must be a string or nonempty list of strings")


def annotate_samples(samples_by_task: dict, provenance: dict) -> None:
    """Add a namespaced contract while preserving all original harness fields.

    correctness_metric explicitly opts into binary 0/1 interpretation; absent
    correctness stays unknown even when another score happens to equal 1.
    """
    for task, protocol in provenance.get('protocols', {}).items():
        for sample in samples_by_task.get(task, []):
            metric = protocol.get('correctness_metric')
            value = sample.get(metric) if metric else None
            if metric and (not isinstance(value, (bool, int, float)) or value not in (0, 1)):
                raise ValueError(f"{task}: correctness_metric must emit boolean or binary 0/1")
            sample['_eval_framework'] = {
                'sample_id': sample['doc'][protocol['sample_id']],
                'primary_filter': protocol['primary_filter'],
                'is_correct': bool(value) if metric else None,
            }
