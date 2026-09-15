"""사용자 관측 함수와 recorder 사이의 계약. 모델 출력은 교체하지 않는다."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import re
from typing import Any, Callable, Sequence

from .storage import ColumnSpec, TableSpec


def first_input(args, kwargs):
    """기본 입력 추출: 첫 positional tensor 또는 hidden_states keyword."""
    return args[0] if args else kwargs.get("hidden_states")


def first_output(output):
    """기본 출력 추출: tensor 또는 tuple/list의 첫 원소."""
    return output[0] if isinstance(output, (tuple, list)) else output


@dataclass(frozen=True)
class HookContext:
    """한 문서의 한 모듈 호출. forward는 기존 ForwardContext이다.

    observer에 전달되는 두 tensor는 [선택된 토큰 수, features]이며,
    각 행은 forward.steps, forward.positions와 대응한다.
    evaluation의 stage는 scoring/prefill/decode이다. collection은 기본 스키마에서
    replay, 확장 스키마에서 loglikelihood/teacher_forced로 기록한다.
    pass_name은 evaluation/collection이다.
    """
    forward: Any
    module_path: str
    call_index: int
    stage: str
    pass_name: str


@dataclass(frozen=True)
class HookSpec:
    """관측 템플릿 하나를 등록한다.

    modules: 모델 기준 정확한 경로들. boundary가 정의한 catalogue에서 선택.
    boundary: decoder_descendant(기본), decoder 또는 lm_head. root container는 분석하지 않는다.
    positions: scored(기본) 또는 full_prompt. 확장 위치에는 nullable step과 입력 token 귀속을 기록.
    doc_ids / position_range: 문서 ID 필터와 post-truncation 입력 좌표 [start, stop).
    max_rows / max_tensor_bytes: hook별 행/파일 bytes 상한. collection resume은 단위별 상한.
    raw_tensors: input/output 중 저장할 선택 복사본. 원 dtype을 safetensors로 보존.
    collection_resume: 모든 collection hook이 함께 활성화해야 하는 custom-only commit 계약.
    module_pattern: 경로에 `re.search`로 맞추는 정규식. `--debug-modules`와 같은 규칙이다.
    select_modules(catalogue): `(경로, 모듈 클래스 이름)` 쌍의 정렬된 tuple을 받아
        선택한 경로들을 반환하는 함수. 살아 있는 모듈 객체는 넘기지 않는다.
        modules·module_pattern·select_modules 중 정확히 하나만 지정한다. 어느 쪽이든
        경로 집합은 모델이 만들어진 뒤 첫 forward 전에 확정되고, 정렬·중복 제거해
        provenance에 남으며, 하나도 맞지 않으면 실행을 실패시킨다.
    observe(inputs, outputs, context): 이름 -> [선택 토큰 수] 숫자 tensor/list.
        입력과 출력은 각각 detach한 독립 복사본이다. 반환값은 저장에만 사용.
        입력·출력의 feature 수는 달라도 되며 비교 방법은 사용자 함수가 정한다.
    metrics: 반환할 지표 이름 -> 설명. 모든 지표는 float64로 저장한다.
    extract_input(args, kwargs), extract_output(output): [batch, sequence, features]
        tensor를 고르는 읽기 전용 함수. 원본을 받으므로 수정하면 안 된다.
        kwargs/cache 등 전체 객체를 복사하지 않고 선택한 tensor만 복사한다.
    version: 의존 코드/설정의 의미가 바뀌면 반드시 변경할 구현 식별자.
    source_files: 함수가 정의된 파일 이외의 로컬 의존 파일. 내용 hash를 기록.
    pass_name: evaluation 또는 collection. collection은 후속 재실행 데이터.

    호출마다 선택된 위치만 복사하고 즉시 축약한다. observer가 tensor를 외부에
    보관하면 메모리 회수는 보장할 수 없다. 임의 Python 코드의 격리는 제공하지 않는다.
    """
    name: str
    # modules 이후가 기본값을 갖는 것은 selector 셋 중 하나만 주기 위해서다. 없으면
    # TypeError 대신 __post_init__이 무엇이 빠졌는지 말한다.
    modules: Sequence[str] = ()
    observe: Callable | None = None
    metrics: dict[str, str] | None = None
    version: str = ""
    extract_input: Callable = first_input
    extract_output: Callable = first_output
    pass_name: str = "evaluation"
    source_files: Sequence[str] = ()
    module_pattern: str | None = None
    select_modules: Callable | None = None
    boundary: str = "decoder_descendant"
    positions: str = "scored"
    doc_ids: Sequence[int] | None = None
    position_range: tuple[int, int] | None = None
    max_rows: int = 100000
    raw_tensors: Sequence[str] = ()
    max_tensor_bytes: int = 67108864
    collection_resume: bool = False

    @property
    def extended(self):
        """Use position-aware schema only when explicitly requested."""
        return self.positions != "scored" or bool(self.raw_tensors) or self.collection_resume


    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", self.name):
            raise ValueError(f"invalid hook name: {self.name!r}")
        if not isinstance(self.version, str) or not self.version:
            raise ValueError(f"hook {self.name}: a version string is required")
        # 세 가지 selector는 서로 배타적이다. 둘을 함께 받으면 어느 쪽이 실제 대상인지
        # provenance만 보고는 알 수 없고, 하나도 없으면 관측할 모듈이 없다.
        given = [field for field, present in (
            ("modules", bool(self.modules)),
            ("module_pattern", self.module_pattern is not None),
            ("select_modules", self.select_modules is not None)) if present]
        if len(given) != 1:
            raise ValueError(f"hook {self.name}: give exactly one of modules, module_pattern "
                             f"or select_modules; got {given or ['none']}")
        if self.modules and (isinstance(self.modules, str)
                             or any(not isinstance(p, str) or not p for p in self.modules)
                             or len(set(self.modules)) != len(self.modules)):
            raise ValueError(f"hook {self.name}: distinct module paths are required")
        if self.module_pattern is not None:
            if not isinstance(self.module_pattern, str):
                raise ValueError(f"hook {self.name}: module_pattern must be a string")
            try:
                re.compile(self.module_pattern)
            except re.error as error:
                raise ValueError(f"hook {self.name}: module_pattern is not a valid regular "
                                 f"expression: {error}") from error
        if self.select_modules is not None and not callable(self.select_modules):
            raise ValueError(f"hook {self.name}: select_modules must be callable")
        if any(k not in ("input", "output") for k in self.raw_tensors) or len(set(self.raw_tensors)) != len(self.raw_tensors):
            raise ValueError("raw_tensors must select distinct input/output tensors")
        if self.max_tensor_bytes < 0:
            raise ValueError("max_tensor_bytes must be nonnegative")
        if self.collection_resume and self.pass_name != "collection":
            raise ValueError("collection_resume requires pass_name=collection")
        if self.positions not in ("scored", "full_prompt"):
            raise ValueError("positions must be scored or full_prompt")
        if self.max_rows < 0:
            raise ValueError("max_rows must be nonnegative")
        if self.doc_ids is not None and any(not isinstance(i, int) or i < 0 for i in self.doc_ids):
            raise ValueError("doc_ids must be nonnegative integers")
        if self.position_range is not None and (len(self.position_range) != 2 or
                not 0 <= self.position_range[0] <= self.position_range[1]):
            raise ValueError("position_range must be a nonnegative [start, stop) pair")
        if self.boundary not in ("decoder_descendant", "decoder", "lm_head"):
            raise ValueError("custom boundary must be decoder_descendant, decoder or lm_head")
        if self.pass_name not in ("evaluation", "collection"):
            raise ValueError(f"hook {self.name}: invalid pass_name")
        if not isinstance(self.metrics, dict) or not self.metrics or any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", k) for k in self.metrics):
            raise ValueError(f"hook {self.name}: metrics must have valid names")
        if any(not isinstance(v, str) for v in self.metrics.values()):
            raise ValueError(f"hook {self.name}: metric descriptions must be strings")
        reserved = {"task_name", "doc_id", "choice_idx", "step", "module_path", "call_index", "stage", "pass_name"}
        if self.extended:
            reserved.update({"input_offset", "forward_index", "position", "input_token_id", "input_length", "token_role"})
        if reserved.intersection(self.metrics):
            raise ValueError(f"hook {self.name}: metric name collides with a context column")
        if not all(callable(f) for f in (self.observe, self.extract_input, self.extract_output)):
            raise ValueError(f"hook {self.name}: callbacks must be callable")

    def resolve(self, catalogue: Sequence[tuple[str, str]]) -> tuple[str, ...]:
        """selector를 실제 경로 집합으로 확정한다. 정렬·중복 제거하며 미매칭은 실패다.

        catalogue는 `(경로, 모듈 클래스 이름)` 쌍이며 호출 시점에 모델이 존재해야 한다.
        결과가 비면 조용히 아무것도 관측하지 않는 대신 실행을 멈춘다. 오타 난 정규식과
        "이 모델에는 그런 모듈이 없다"는 사실은 둘 다 관측 0건으로 끝나기 때문이다.
        """
        known = {path for path, _ in catalogue}
        if self.modules:
            missing = [path for path in self.modules if path not in known]
            if missing:
                raise ValueError(f"hook {self.name}: {missing[0]!r} is not a decoder descendant module")
            chosen: list[str] = list(self.modules)
            described = "the given module paths"
        elif self.module_pattern is not None:
            pattern = re.compile(self.module_pattern)
            chosen = [path for path, _ in catalogue if pattern.search(path)]
            described = f"module_pattern {self.module_pattern!r}"
        else:
            try:
                chosen = list(self.select_modules(tuple(catalogue)))
            except Exception as exc:
                raise ValueError(f"hook {self.name}: select_modules failed: {exc}") from exc
            if any(not isinstance(path, str) for path in chosen):
                raise ValueError(f"hook {self.name}: select_modules must return module paths as strings")
            unknown = [path for path in chosen if path not in known]
            if unknown:
                raise ValueError(f"hook {self.name}: select_modules returned {unknown[0]!r}, "
                                 "which is not a decoder descendant module")
            described = "select_modules"
        if not chosen:
            raise ValueError(f"hook {self.name}: {described} matched none of the "
                             f"{len(catalogue)} decoder submodules")
        return tuple(sorted(set(chosen)))

    def selector(self) -> dict[str, Any] | None:
        """provenance에 남길 selector 식별. exact path는 기존 modules가 곧 selector다."""
        if self.module_pattern is not None:
            return {"kind": "pattern", "pattern": self.module_pattern,
                    "match": "re.search on the dotted module path"}
        if self.select_modules is not None:
            return {"kind": "function",
                    "function": getattr(self.select_modules, "__qualname__",
                                        type(self.select_modules).__qualname__),
                    "match": "returns paths from (path, module type) pairs"}
        return None

    def table_spec(self) -> TableSpec:
        """실행별 저장 schema. 전역 TABLES를 수정하지 않는다."""
        import pyarrow as pa
        keys = ("task_name", "doc_id", "choice_idx", "step", "module_path", "call_index", "stage", "pass_name")
        columns = tuple(ColumnSpec(k, pa.int64() if k in {"doc_id", "choice_idx", "step", "call_index"} else pa.string(), k) for k in keys)
        if self.extended:
            keys = tuple(k for k in keys if k != "step") + ("forward_index", "position")
            columns += tuple(ColumnSpec(k, pa.int64(), k) for k in
                             ("forward_index", "position", "input_token_id", "input_length", "input_offset"))
            columns += (ColumnSpec("token_role", pa.string(), "Role of the actual input token"),)
        columns += tuple(ColumnSpec(k, pa.float64(), v, comparable_across_vocab=False) for k, v in self.metrics.items())
        return TableSpec(f"custom/{self.name}", keys, columns, None, f"Custom observation: {self.name}")

    def descriptor(self) -> dict[str, Any]:
        """소스 파일과 명시한 의존 파일의 내용을 실행 식별에 포함한다.

        source를 읽을 수 없는 메모리 함수는 version에 의존하며 자동 resume을
        허용하지 않는다. 패키지/외부 의존성 변경은 version에 반영해야 한다.
        """
        files = set(self.source_files)
        complete = True
        callbacks = [self.observe, self.extract_input, self.extract_output]
        # 선택 함수도 같은 규칙을 따른다. 소스를 읽을 수 없으면 resume_safe가 아니다.
        for fn in callbacks + ([self.select_modules] if self.select_modules is not None else []):
            try:
                path = inspect.getsourcefile(fn)
            except TypeError:
                path = None
            if path and Path(path).is_file():
                files.add(path)
            else:
                complete = False
        sources = {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted(files)}
        # selector 키는 정규식·함수를 쓸 때만 붙인다. exact path hook의 descriptor는
        # 이 변경 전과 같아야 이전 실행의 후속 수집이 계속 복원된다.
        selector = self.selector()
        return {"name": self.name, "modules": list(self.modules),
                **({"selector": selector} if selector else {}),
                **({"boundary": self.boundary, "input_layout": "BS_token_ids_or_BSF" if self.boundary == "decoder" else "BSF",
                    "output_features": "vocabulary" if self.boundary == "lm_head" else "hidden"}
                   if self.boundary != "decoder_descendant" else {}), "metrics": self.metrics,
                "version": self.version, "pass_name": self.pass_name, "layout": "BSF",
                "positions": self.positions,
                **({"doc_ids": list(self.doc_ids) if self.doc_ids is not None else None,
                    "position_range": list(self.position_range) if self.position_range is not None else None, "max_rows": self.max_rows,
                    "position_contract": "post_truncation_input; cached_decode_only; nullable_scored_step",
                    "raw_tensors": list(self.raw_tensors), "max_tensor_bytes": self.max_tensor_bytes,
                    "tensor_dtype": "original", "collection_resume": self.collection_resume}
                   if self.extended or self.doc_ids is not None or self.position_range is not None or self.max_rows != 100000 else {}), "sources": sources, "resume_safe": complete,
                "callbacks": [getattr(f, "__qualname__", type(f).__qualname__) for f in callbacks],
                "schema": json.loads(self.table_spec().arrow_schema().metadata[b"eval_framework"])}


def load_hooks(factory: str | None, config: dict, hooks: Sequence[HookSpec] = ()) -> list[HookSpec]:
    """Python 객체 또는 module:function(config) factory를 같은 API로 검증한다."""
    if not isinstance(config, dict):
        raise ValueError("hook config must be a JSON object")
    result = list(hooks)
    if factory:
        module, sep, name = factory.partition(":")
        if not sep or not module or not name:
            raise ValueError("hook factory must be package.module:function")
        try:
            fn = getattr(importlib.import_module(module), name)
            built = fn(config)
            result.extend(built)
        except Exception as exc:
            raise ValueError(f"hook factory {factory}: {exc}") from exc
    if any(not isinstance(h, HookSpec) for h in result):
        raise ValueError("hook factory must return a sequence of HookSpec")
    if len({h.name for h in result}) != len(result):
        raise ValueError("duplicate custom hook name")
    return result


def ensure_collection_is_empty(run_dir, specs) -> None:
    """완료 근거 없는 legacy Parquet는 재개하지 않는다. opt-in transaction만 별도 경로를 쓴다."""
    collection = [s for s in specs if s.pass_name == "collection"]
    if any(s.collection_resume for s in collection) and not all(s.collection_resume for s in collection):
        raise ValueError("all collection hooks must opt into collection_resume together")
    for spec in specs:
        if spec.pass_name == "collection" and any(Path(run_dir, "custom", spec.name).glob("*.parquet")):
            raise ValueError("custom collection data already exists; use a new run directory")


class HookRuntime:
    """한 recorder session의 관측 lifecycle. 입력 참조는 호출 종료 시 해제한다."""
    def __init__(self, recorder, specs, pass_name):
        self.recorder = recorder
        self.specs = [s for s in specs if s.pass_name == pass_name]
        self.pass_name = pass_name
        self.handles = []
        self.pending = {}
        self.calls = {}
        self.rows = {}
        self.active = False
        self.resume_checked = False
        # 이름 -> 확정된 경로. 한 모델에 한 번만 확정하고 session마다 다시 풀지 않는다.
        self.targets: dict[str, tuple[str, ...]] = {}
        self.baseline_checked = False
        self.forward_contexts = None
        self.suspended = False
        self.input_ids = None
        self.logits_to_keep = 0
        self.total_rows = {}
        self.tensor_bytes = {}
        self.tensor_indexes = []
        self.attempt = None
        self.collection = None
        self.unit_rows = {}
        self.unit_coverage = {}

    def register(self) -> None:
        """대상·resume 검증을 마친 뒤 hook을 부착하고 필요한 저장소를 연다.

        부착한 handle은 즉시 목록에 넣는다. 이후 다른 hook이나 저장소 등록이
        실패해도 close가 부분 등록된 handle을 빠짐없이 회수할 수 있게 한다.
        """
        ensure_collection_is_empty(self.recorder.writer.run_dir, self.specs)
        modules = self._registration_modules()
        # selector·이전 기록 검사는 hook 부착 전에 끝낸다.
        self.resolve(modules)
        self.check_resolved_set()
        self.check_resume_rows()
        try:
            for spec in self.specs:
                self.recorder.writer.register_table(spec.table_spec())
                for path in self.targets[spec.name]:
                    pre, post = self.callbacks(spec, path)
                    self.handles.append(modules[path].register_forward_pre_hook(pre, with_kwargs=True))
                    self.handles.append(modules[path].register_forward_hook(post, with_kwargs=True))
            if self.specs and self.specs[0].collection_resume:
                self._open_collection_store()
        except BaseException:
            self.close()
            raise

    def _registration_modules(self) -> dict[str, Any]:
        """모델 기준 모듈 경로를 만들고 boundary별 selector 허용 범위를 기록한다.

        decoder.named_modules()의 경로는 decoder 기준이다. block 경로의 prefix를
        붙여 모델 기준으로 바꾼다. decoder 자체와 lm_head는 각각 별도 boundary에
        넣어 descendant selector가 실수로 root까지 관측하지 않게 한다.
        """
        prefix = self.recorder.adapter.paths.blocks.rpartition(".")[0]
        modules = {f"{prefix}.{name}" if prefix else name: module
                   for name, module in self.recorder.adapter.decoder.named_modules() if name}
        root = self.recorder.adapter.root_model
        if any(s.boundary != "decoder_descendant" for s in self.specs) and root is None:
            raise ValueError("decoder/lm_head boundaries require adapter.root_model")
        if any(s.boundary == "lm_head" for s in self.specs) and not self.recorder.adapter.lm_head_path:
            from .adapters import _resolve_named
            path, _ = _resolve_named(root, self.recorder.adapter.paths.lm_head)
            self.recorder.adapter.lm_head_path = path
        self.boundary_paths = {"decoder_descendant": set(modules), "decoder": {prefix},
                               "lm_head": {self.recorder.adapter.lm_head_path}}
        modules[prefix] = self.recorder.adapter.decoder
        if root is not None:
            modules.update({p: m for p, m in root.named_modules()
                            if p == self.recorder.adapter.lm_head_path})
        elif any(s.boundary == "lm_head" for s in self.specs):
            raise ValueError("lm_head boundary requires adapter.root_model")
        return modules

    def _open_collection_store(self) -> None:
        """재개 가능한 custom 수집의 계약을 확인하고 저장소를 연다.

        register의 예외 처리 안에서 호출한다. 계약 검증이나 저장소 생성이 실패해도
        이미 붙인 hook은 register가 회수한다. 모델·샘플·관측 경로의 식별 정보는
        이전 수집과 같은 데이터를 이어 쓰는지 확인하기 위한 계약에 포함된다.
        """
        if self.recorder.reducers:
            raise ValueError("custom collection_resume requires custom-only collection reducers; legacy dumps have no shared commit contract")
        from .storage import CustomCollectionStore, read_manifest
        ensure_collection_is_empty(self.recorder.writer.run_dir, self.specs)
        descriptors = [s.descriptor() for s in self.specs]
        if any(not d["resume_safe"] for d in descriptors):
            raise ValueError("custom collection resume requires source-backed callbacks")
        root = Path(self.recorder.writer.run_dir)
        identity = None
        if (root / "results.json").exists():
            manifest = read_manifest(root)
            identity = manifest.get("config_identity")
            if not identity:
                raise ValueError("incomplete custom collection manifest: config_identity missing")
        samples_path = root / "samples.jsonl"
        contract = {
            "version": 1,
            "specs": descriptors,
            "resolved": self.resolved_paths(),
            "config_identity": identity,
            "samples_sha256": (
                hashlib.sha256(samples_path.read_bytes()).hexdigest()
                if samples_path.is_file() else None
            ),
        }
        self.collection = CustomCollectionStore(root, contract)

    def resolve(self, modules: dict[str, Any]) -> None:
        """모델이 존재하는 첫 시점에 경로 집합을 확정한다. session마다 바뀌지 않는다.

        사용자 선택 함수에는 살아 있는 모듈이 아니라 `(경로, 클래스 이름)` 쌍만 넘긴다.
        관측 계약은 모델을 바꾸지 않는 것이고, 선택 단계라고 다르지 않다.
        """
        if self.targets:
            return
        catalogue = tuple(sorted((path, type(module).__name__) for path, module in modules.items()))
        targets = {}
        for spec in self.specs:
            allowed_paths = self.boundary_paths[spec.boundary]
            candidates = tuple(item for item in catalogue if item[0] in allowed_paths)
            targets[spec.name] = spec.resolve(candidates)
        # 모든 selector가 성공한 뒤 한 번에 확정한다. 중간 실패로 일부만 남으면
        # 다음 등록의 `if self.targets`가 미완료 상태를 완료로 오인할 수 있다.
        self.targets = targets

    def check_resolved_set(self):
        """같은 디렉터리가 기록한 경로 집합과 달라지면 섞지 않는다.

        selector 자체는 실행 식별에 들어가므로 정규식을 바꾸면 resume이 먼저 거부된다.
        여기서 잡는 것은 selector가 그대로인 채 모델 구조가 달라져 같은 이름의 hook이
        다른 모듈을 관측하게 되는 경우다. 별도 프로세스의 후속 수집도 같은 검사를 지난다.
        """
        if self.baseline_checked or not self.specs:
            return
        self.baseline_checked = True
        from .storage import read_manifest
        try:
            recorded = read_manifest(self.recorder.writer.run_dir).get("custom_hooks") or {}
        except (OSError, ValueError, KeyError):
            return
        for spec in self.specs:
            before = (recorded.get("resolved") or {}).get(spec.name)
            now = list(self.targets[spec.name])
            if before is None or list(before) == now:
                continue
            changed = sorted(set(before).symmetric_difference(now))
            raise ValueError(
                f"hook {spec.name}: the selector now resolves to {len(now)} modules where this "
                f"run recorded {len(before)}, differing at {changed[0]!r}; use a new run directory")

    def resolved_paths(self) -> dict[str, list[str]]:
        """provenance에 기록할 이름 -> 실제 경로. 확정 전에는 비어 있다."""
        return {name: list(paths) for name, paths in self.targets.items()}

    def check_resume_rows(self):
        """steps만 먼저 flush된 중단 실행을 완료된 custom 관측으로 취급하지 않는다."""
        if self.resume_checked or not self.specs or not self.recorder.already_recorded:
            return
        if any(s.extended for s in self.specs):
            raise ValueError("extended evaluation custom capture cannot resume from steps alone; use a new run directory or standalone collection resume")
        from .storage import read_table
        run_dir = self.recorder.writer.run_dir
        keys = ["task_name", "doc_id", "choice_idx", "step"]
        steps = read_table(run_dir, "steps")
        expected = set(steps[keys].itertuples(index=False, name=None))
        for spec in self.specs:
            try:
                frame = read_table(run_dir, f"custom/{spec.name}")
            except FileNotFoundError as exc:
                raise ValueError(f"hook {spec.name}: missing custom data for recorded steps; use a new run directory") from exc
            for path in self.targets[spec.name]:
                observed = set(frame.loc[frame.module_path == path, keys].itertuples(index=False, name=None))
                if expected != observed:
                    raise ValueError(f"hook {spec.name}: incomplete custom data for recorded steps; use a new run directory")
        self.resume_checked = True

    def note_root(self, module, args, kwargs):
        """Only explicit integer logits_to_keep establishes a trailing lm_head layout."""
        self.logits_to_keep = kwargs.get("logits_to_keep", kwargs.get("num_logits_to_keep", 0))

    def begin(self, args=(), kwargs=None):
        self.active = self.recorder._plan is not None
        kwargs = kwargs or {}
        ids = args[0] if args else kwargs.get("input_ids")
        # Retain only during this forward. No raw input reference survives end/close.
        self.input_ids = ids if getattr(ids, "ndim", None) == 2 else None
        self.calls.clear()
        self.pending.clear()
        self.forward_contexts = None

    def contexts(self, length):
        """같은 forward의 모든 hook에 동일한 문서·step 매핑을 제공한다.

        recorder의 context 생성은 generation step을 전진시킨다. 여기서는 값을
        복원하고, 실제 전진은 decoder 종료 시 recorder가 한 번만 수행하게 한다.
        """
        if self.forward_contexts is not None:
            return self.forward_contexts
        plan = self.recorder._plan
        step = plan.get("next_step")
        contexts = self.recorder._contexts_for_forward(length)
        if step is not None:
            plan["next_step"] = step
        self.forward_contexts = contexts
        return contexts

    def _head_sequence_offset(self, spec: HookSpec, length: int) -> int:
        """입력 tensor 좌표를 마지막 토큰만 남긴 lm_head의 좌표로 옮길 offset.

        logits_to_keep가 명시한 trailing slice만 지원한다. 길이가 같아도 index
        tensor는 토큰 순서를 바꿀 수 있으므로 정수 여부를 먼저 검사한다.
        """
        if spec.boundary != "lm_head":
            return 0
        keep = self.logits_to_keep
        if type(keep) is not int or keep < 0:
            raise ValueError("custom lm_head: unsupported logits_to_keep; requires a nonnegative integer")
        if self.input_ids is None or length == self.input_ids.shape[1]:
            return 0
        width = self.input_ids.shape[1]
        if keep <= 0 or length != min(keep, width):
            raise ValueError("custom lm_head: unsupported sequence layout (requires explicit integer logits_to_keep)")
        if spec.positions == "full_prompt":
            raise ValueError("custom lm_head full_prompt unavailable: logits_to_keep omits prompt positions")
        return width - length

    def selected_contexts(self, spec, length):
        """Map actual input coordinates independently of nullable scoring steps.

        Input positions start at zero after truncation. Incremental decode has one
        physical position and an absolute post-truncation sequence coordinate.
        Prefix recomputation is rejected until an explicit forward schema supports it.
        """
        result = []
        for ctx in self.contexts(length):
            if spec.doc_ids is not None and ctx.doc_id not in spec.doc_ids:
                continue
            scored_steps = dict(zip(ctx.positions, ctx.steps))
            target_tokens = (dict(zip(ctx.positions, ctx.target_token_ids))
                             if ctx.target_token_ids is not None else None)
            plan = self.recorder._plan
            incremental = plan["kind"] == "generate"
            forward_index = ctx.steps[0] if incremental else 0
            if incremental and forward_index > 0 and (self.input_ids.shape[1] if self.input_ids is not None else length) != 1 and spec.extended:
                raise ValueError("custom full_prompt: cache-free prefix recomputation is unsupported")
            if spec.positions == "full_prompt":
                if self.input_ids is None:
                    raise ValueError("custom full_prompt requires actual input_ids")
                if incremental and forward_index > 0:
                    positions, indices = list(ctx.positions), list(ctx.seq_indices)
                else:
                    valid = ctx.input_length if ctx.input_length is not None else (length if incremental else None)
                    if valid is None:
                        raise ValueError("custom full_prompt requires input_length")
                    offset = ctx.seq_indices[0] - ctx.positions[0]
                    positions = list(range(valid))
                    indices = [p + offset for p in positions]
            else:
                positions, indices = list(ctx.positions), list(ctx.seq_indices)
            head_offset = self._head_sequence_offset(spec, length)
            selected = [(p, i) for p, i in zip(positions, indices)
                        if spec.position_range is None or spec.position_range[0] <= p < spec.position_range[1]]
            if not selected:
                continue
            positions, indices = map(list, zip(*selected))
            prompt_end = (plan["prompt_length"] if incremental else min(ctx.positions) + 1)
            roles = [("prompt" if p < prompt_end else
                      "continuation" if ctx.task_kind == "loglikelihood" else "generated") for p in positions]
            token_ids = ([int(self.input_ids[ctx.batch_row, i]) for i in indices]
                         if self.input_ids is not None else [None] * len(indices))
            metadata = {"forward_index": forward_index, "token_roles": roles, "input_token_ids": token_ids,
                        "input_length": ctx.input_length or (plan.get("prompt_length", length) + forward_index),
                        "input_offset": ctx.input_offset}
            # token ID는 원래 입력 좌표에서 읽고, head offset은 관측 tensor를 고를 때만
            # 적용한다. 채점 대상이 아닌 prompt 위치의 step/target은 None으로 둔다.
            result.append(replace(ctx, steps=[scored_steps.get(p) for p in positions], positions=positions,
                                  seq_indices=[i - head_offset for i in indices],
                                  target_token_ids=([target_tokens.get(p) for p in positions]
                                                    if target_tokens is not None else None),
                                  step_token_ids=None,
                                  shared={"custom": metadata}))
        return result

    @staticmethod
    def select(tensor, ctx):
        import torch
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
            raise ValueError("custom hook extractor must return a [batch, sequence, features] tensor")
        if ctx.batch_row >= tensor.shape[0] or any(i < 0 or i >= tensor.shape[1] for i in ctx.seq_indices):
            raise ValueError("custom hook layout does not match the request positions")
        return tensor[ctx.batch_row, ctx.seq_indices, :].detach().clone()

    def callbacks(self, spec, path):
        """입력 복사본과 출력을 호출별로 짝지어 관측하는 PyTorch hook 쌍.

        pending은 stack이다. 같은 모듈이 중첩 호출되어도 마지막 pre-hook의
        입력부터 대응시킨다. 두 callback은 None을 반환해 모델 값을 유지한다.
        """
        key = (spec.name, path)

        def pre(module, args, kwargs):
            if not self.active or self.suspended:
                return
            if spec.boundary == "decoder" and spec.extract_input is first_input:
                tensor = args[0] if args else kwargs.get("inputs_embeds")
                if tensor is None:
                    tensor = kwargs.get("input_ids")
                if getattr(tensor, "ndim", None) == 2:
                    tensor = tensor.unsqueeze(-1)
            else:
                tensor = spec.extract_input(args, kwargs)
            # context를 만들기 전에 shape을 검증한다.
            if getattr(tensor, "ndim", None) != 3:
                raise ValueError(f"hook {spec.name} at {path}: input must have BSF layout")
            contexts = self.selected_contexts(spec, tensor.shape[1])
            self._check_row_limit(spec, sum(ctx.n_positions for ctx in contexts))
            call = self.calls.get(key, 0)
            self.calls[key] = call + 1
            coverage = self.unit_coverage.setdefault(spec.name, {})
            coverage[path] = coverage.get(path, 0) + 1
            self.pending.setdefault(key, []).append((call, [(ctx, self.select(tensor, ctx)) for ctx in contexts]))

        def post(module, args, kwargs, output):
            if not self.active or self.suspended:
                return
            import torch
            call, inputs = self.pending[key].pop()
            tensor = (getattr(output, "last_hidden_state", None)
                      if spec.boundary == "decoder" and spec.extract_output is first_output
                      else spec.extract_output(output))
            try:
                for ctx, before in inputs:
                    stage = self._stage(spec, ctx)
                    context = HookContext(ctx, path, call, stage, self.pass_name)
                    if spec.raw_tensors:
                        size = (before.numel() * before.element_size() if "input" in spec.raw_tensors else 0)
                        if "output" in spec.raw_tensors:
                            size += ctx.n_positions * tensor.shape[-1] * tensor.element_size()
                        if self.tensor_bytes.get(spec.name, 0) + size > spec.max_tensor_bytes:
                            raise ValueError(f"custom max_tensor_bytes exceeded: {spec.name}")
                    after = self.select(tensor, ctx)
                    self._check_row_limit(spec, ctx.n_positions)
                    self.total_rows[spec.name] = self.total_rows.get(spec.name, 0) + ctx.n_positions
                    # Save before invoking user code: observers may mutate their independent copies.
                    if spec.raw_tensors:
                        self.save_tensors(spec, before, after, context)
                    with torch.no_grad():
                        values = spec.observe(before, after, context)
                    scalars = self._metric_scalars(spec, values, ctx.n_positions)
                    for i in range(ctx.n_positions):
                        row = dict(ctx.axis_columns(i), module_path=path, call_index=call, stage=stage, pass_name=self.pass_name)
                        if spec.extended:
                            info = ctx.shared["custom"]
                            row.update(position=ctx.positions[i], forward_index=info["forward_index"],
                                       input_token_id=info["input_token_ids"][i], token_role=info["token_roles"][i],
                                       input_length=info["input_length"], input_offset=info["input_offset"])
                        row.update({name: values[i] for name, values in scalars.items()})
                        self.rows.setdefault(f"custom/{spec.name}", []).append(row)
            except Exception as exc:
                raise RuntimeError(f"hook {spec.name} at {path}, call {call}: {exc}") from exc
            # PyTorch에는 항상 None을 반환하여 모델 출력을 유지한다.
        return pre, post

    def _check_row_limit(self, spec: HookSpec, additional_rows: int) -> None:
        """기존 기본 스키마의 무제한 동작을 유지하고 opt-in 상한만 검사한다.

        pre-hook에서는 입력 복사 전에 전체 선택을 검사하고, post-hook에서는
        중첩 호출이 추가한 행까지 포함해 실제 기록 직전에 다시 검사한다.
        """
        enforce_limit = spec.extended or spec.max_rows != 100000
        if enforce_limit and self.total_rows.get(spec.name, 0) + additional_rows > spec.max_rows:
            raise ValueError(f"custom max_rows exceeded: {spec.name}")

    def _stage(self, spec: HookSpec, ctx: Any) -> str:
        """실제 실행 단계의 저장 이름. 기본 collection의 replay 이름도 보존한다."""
        if self.pass_name == "collection":
            if not spec.extended:
                return "replay"
            return "loglikelihood" if ctx.task_kind == "loglikelihood" else "teacher_forced"
        if ctx.task_kind == "loglikelihood":
            return "scoring"
        return "prefill" if ctx.shared["custom"]["forward_index"] == 0 else "decode"

    @staticmethod
    def _metric_scalars(spec: HookSpec, values: Any, n_positions: int) -> dict[str, list[float]]:
        """observer 결과를 검증하고 저장용 CPU 숫자로 바꿔 activation 참조를 끊는다.

        각 지표는 선택한 토큰마다 실수 하나를 반환해야 한다. 모든 지표를 검증한
        뒤에 행을 만들므로 뒤쪽 지표가 잘못되어도 일부 지표만 기록되지 않는다.
        """
        import torch

        if not isinstance(values, dict) or set(values) != set(spec.metrics):
            raise ValueError("returned metric names do not match HookSpec.metrics")
        scalars = {}
        for name, value in values.items():
            value = torch.as_tensor(value).detach()
            if tuple(value.shape) != (n_positions,) or value.is_complex():
                raise ValueError(f"metric {name} must have shape [{n_positions}] and be real-valued")
            scalars[name] = value.to(device="cpu", dtype=torch.float64).tolist()
        return scalars

    def save_tensors(self, spec, before, after, context):
        """Persist a bounded selection in its original dtype, indexed without extension code."""
        from .storage import save_custom_tensors
        selected = {k: (before if k == "input" else after) for k in spec.raw_tensors}
        used = self.tensor_bytes.get(spec.name, 0)
        size = sum(t.numel() * t.element_size() for t in selected.values())
        if used + size > spec.max_tensor_bytes:
            raise ValueError(f"custom max_tensor_bytes exceeded: {spec.name}")
        ctx = context.forward
        info = ctx.shared["custom"]
        rows = [dict(ctx.axis_columns(i), position=ctx.positions[i],
                     forward_index=info["forward_index"], module_path=context.module_path,
                     call_index=context.call_index, stage=context.stage, pass_name=self.pass_name,
                     input_token_id=info["input_token_ids"][i], token_role=info["token_roles"][i],
                     input_offset=info["input_offset"])
                for i in range(ctx.n_positions)]
        index, actual_size = save_custom_tensors(self.recorder.writer.run_dir, spec.name, selected,
            rows, "vocabulary" if spec.boundary == "lm_head" else "hidden", self.attempt,
            spec.max_tensor_bytes - used, input_features="token_id" if spec.boundary == "decoder" and before.shape[-1] == 1 else "hidden")
        self.tensor_bytes[spec.name] = used + actual_size
        self.tensor_indexes.append(index)

    def end(self):
        self.active = False
        self.input_ids = None
        self.forward_contexts = None
        if self.recorder._plan is not None:
            for spec in self.specs:
                for path in self.targets.get(spec.name, ()):
                    if not spec.collection_resume and (spec.name, path) not in self.calls:
                        raise RuntimeError(f"hook {spec.name}: target {path} was not called in this forward")
        self.pending.clear()

    def begin_unit(self, task_name, doc_id, choice_idx):
        """Skip only a proven commit; otherwise allocate a fresh, non-overwriting attempt."""
        if self.collection is None:
            return True
        import uuid
        self.unit_key = (task_name, doc_id, choice_idx, "collection")
        if self.unit_key in self.collection.completed:
            return False
        self.attempt = uuid.uuid4().hex
        self.unit_rows = {}
        self.tensor_indexes = []
        self.total_rows = {}
        self.tensor_bytes = {}
        self.unit_coverage = {s.name: {p: 0 for p in self.targets[s.name]} for s in self.specs}
        return True

    def finish_unit(self):
        """Publish all scalar rows, tensor indexes and explicit zero-call coverage together."""
        if self.collection is None:
            return
        self.collection.commit(self.unit_key, self.attempt, self.unit_rows, self.specs,
                               self.tensor_indexes, self.unit_coverage)
        self.attempt = None
        self.unit_rows = {}
        self.tensor_indexes = []

    def drain(self):
        rows, self.rows = self.rows, {}
        if self.collection is not None:
            for name, values in rows.items():
                self.unit_rows.setdefault(name, []).extend(values)
            return {}
        return rows

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self.collection is not None:
            self.collection.close()
            self.collection = None
        self.unit_rows = {}
        self.tensor_indexes = []
        self.attempt = None
        self.input_ids = None
        self.forward_contexts = None
        self.pending.clear()
        self.calls.clear()
        self.active = False
