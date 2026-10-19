from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(errors="replace"))


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _field_names(row: dict[str, Any]) -> list[str]:
    fields: list[str] = []
    for key in ("fields", "columns", "keys", "required_fields", "required_columns", "required_keys", "sections"):
        for value in _as_list(row.get(key)):
            if isinstance(value, str) and value and value not in fields:
                fields.append(value)
    return fields


def _numeric_fields(row: dict[str, Any]) -> list[str]:
    fields: list[str] = []
    for key in ("numeric_fields", "numeric_columns", "required_numeric_fields", "required_numeric_columns"):
        for value in _as_list(row.get(key)):
            if isinstance(value, str) and value and value not in fields:
                fields.append(value)
    return fields


def _non_empty_fields(row: dict[str, Any]) -> list[str]:
    fields: list[str] = []
    for key in ("non_empty_fields", "non_empty_columns", "required_non_empty_fields", "required_non_empty_columns"):
        for value in _as_list(row.get(key)):
            if isinstance(value, str) and value and value not in fields:
                fields.append(value)
    return fields


def _expected_type_map(row: dict[str, Any]) -> dict[str, str]:
    value = row.get("expected_types") or row.get("field_types") or row.get("column_types") or {}
    return {str(k): str(v).lower() for k, v in value.items()} if isinstance(value, dict) else {}


def _list_item_type_map(row: dict[str, Any]) -> dict[str, str]:
    value = row.get("list_item_types") or row.get("array_item_types") or {}
    return {str(k): str(v).lower() for k, v in value.items()} if isinstance(value, dict) else {}


def _semantic_requirements(row: dict[str, Any]) -> list[str]:
    requirements: list[str] = []
    for key in (
        "semantic_checks",
        "scoring_critical_requirements",
        "scoring_critical_rules",
        "objective_checks",
        "value_checks",
    ):
        for value in _as_list(row.get(key)):
            if isinstance(value, str) and value.strip() and value.strip() not in requirements:
                requirements.append(value.strip())
    return requirements


def _path_value(row: dict[str, Any]) -> str | None:
    for key in ("path", "filename", "artifact", "artifact_path", "artifact_ref", "output_file"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().replace("\\", "/")
    return None


def _hidden_or_forbidden_ref(rel: str) -> bool:
    parts = {part.lower() for part in Path(rel.replace("\\", "/")).parts}
    return "expected" in parts or "hidden" in parts or rel.startswith("/") or ".." in Path(rel).parts


def _candidate_paths(base: Path, rel: str) -> list[Path]:
    rel = rel.lstrip("./")
    paths = [base / rel]
    if rel.startswith("output/"):
        paths.append(base / rel.removeprefix("output/"))
    else:
        paths.append(base / "output" / rel)
        paths.append(base / "artifacts" / rel)
    return list(dict.fromkeys(paths))


def _coerce_number(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(str(value))
    except Exception:
        return None


def _check_json_fields(path: Path, fields: list[str]) -> list[str]:
    try:
        data = _read_json(path)
    except Exception as exc:
        return [f"{path.name} is not valid JSON: {type(exc).__name__}"]
    if not fields:
        return []
    if isinstance(data, dict):
        missing = [field for field in fields if field not in data]
        return [f"{path.name} missing JSON field {field}" for field in missing]
    if isinstance(data, list) and data and isinstance(data[0], dict):
        missing = [field for field in fields if field not in data[0]]
        return [f"{path.name} first row missing JSON field {field}" for field in missing]
    return [f"{path.name} cannot be checked for JSON object fields"]


def _json_data_at_path(path: Path) -> Any:
    try:
        return _read_json(path)
    except Exception:
        return None


def _lookup_path(data: Any, field_path: str) -> tuple[bool, Any]:
    current = data
    for part in str(field_path).split("."):
        if part == "":
            continue
        if isinstance(current, dict):
            if part not in current:
                return False, None
            current = current[part]
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except Exception:
                return False, None
        else:
            return False, None
    return True, current


def _type_matches(value: Any, expected_type: str) -> bool:
    expected_type = expected_type.lower()
    if expected_type in {"number", "numeric", "float"}:
        return _coerce_number(value) is not None
    if expected_type in {"int", "integer"}:
        number = _coerce_number(value)
        return number is not None and float(number).is_integer()
    if expected_type in {"string", "str"}:
        return isinstance(value, str)
    if expected_type in {"boolean", "bool"}:
        return isinstance(value, bool)
    if expected_type in {"array", "list"}:
        return isinstance(value, list)
    if expected_type in {"object", "dict"}:
        return isinstance(value, dict)
    return True


def _check_csv_fields(path: Path, fields: list[str], delimiter: str = ",") -> list[str]:
    try:
        with path.open(newline="", errors="replace") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            header = next(reader, [])
    except Exception as exc:
        return [f"{path.name} is not valid delimited text: {type(exc).__name__}"]
    missing = [field for field in fields if field not in header]
    return [f"{path.name} missing column {field}" for field in missing]


def _csv_rows(path: Path, delimiter: str = ",") -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        header = list(reader.fieldnames or [])
        return header, list(reader)


def _derived_value_failures(base: Path, item: dict[str, Any], rel: str, output_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for check in _as_list(item.get("derived_value_checks")):
        if not isinstance(check, dict):
            continue
        operation = str(check.get("operation") or "").lower()
        if operation != "group_mean_difference":
            continue
        input_csv = check.get("input_csv") or check.get("input_expression_matrix")
        metadata_csv = check.get("metadata_csv") or check.get("input_metadata")
        if not isinstance(input_csv, str) or not isinstance(metadata_csv, str):
            continue
        input_path = next((candidate for candidate in _candidate_paths(base, input_csv) if candidate.exists()), None)
        metadata_path = next((candidate for candidate in _candidate_paths(base, metadata_csv) if candidate.exists()), None)
        if not input_path or not metadata_path:
            failures.append(
                {
                    "path": rel,
                    "kind": "output",
                    "issue": f"derived_check_inputs_missing:{check.get('check_id') or operation}",
                    "repair_hint": "Run the visible-input derived-value check only after input CSV and metadata CSV are present.",
                }
            )
            continue
        try:
            _meta_header, meta_rows = _csv_rows(metadata_path)
            _expr_header, expr_rows = _csv_rows(input_path)
        except Exception:
            continue
        sample_column = str(check.get("sample_column") or "sample")
        group_column = str(check.get("group_column") or "group")
        group_a = str(check.get("group_a") or check.get("normal_group") or "normal")
        group_b = str(check.get("group_b") or check.get("tumor_group") or "tumor")
        key_column = str(check.get("key_column") or check.get("gene_column") or "gene")
        output_key = str(check.get("output_key_column") or key_column)
        output_a = str(check.get("output_group_a_mean_column") or check.get("normal_mean_column") or "mean_normal")
        output_b = str(check.get("output_group_b_mean_column") or check.get("tumor_mean_column") or "mean_tumor")
        output_delta = str(check.get("output_delta_column") or check.get("target_column") or "log2FC")
        tolerance = float(check.get("tolerance") or 1e-3)
        samples_a = [row.get(sample_column) for row in meta_rows if str(row.get(group_column) or "").lower() == group_a.lower()]
        samples_b = [row.get(sample_column) for row in meta_rows if str(row.get(group_column) or "").lower() == group_b.lower()]
        expected_by_key: dict[str, tuple[float, float, float]] = {}
        for expr in expr_rows:
            key = str(expr.get(key_column) or "")
            values_a = [_coerce_number(expr.get(str(sample))) for sample in samples_a if sample]
            values_b = [_coerce_number(expr.get(str(sample))) for sample in samples_b if sample]
            values_a = [value for value in values_a if value is not None]
            values_b = [value for value in values_b if value is not None]
            if not key or not values_a or not values_b:
                continue
            mean_a = sum(values_a) / len(values_a)
            mean_b = sum(values_b) / len(values_b)
            expected_by_key[key] = (mean_a, mean_b, mean_b - mean_a)
        for output in output_rows:
            key = str(output.get(output_key) or "")
            if key not in expected_by_key:
                continue
            mean_a, mean_b, delta = expected_by_key[key]
            checks = [(output_a, mean_a), (output_b, mean_b), (output_delta, delta)]
            for field, expected in checks:
                observed = _coerce_number(output.get(field))
                if observed is None or abs(observed - expected) > tolerance:
                    failures.append(
                        {
                            "path": rel,
                            "kind": "output",
                            "issue": f"derived_value_mismatch:{field}",
                            "repair_hint": f"Recompute {field} in {rel} from visible input files using the declared group-mean difference rule.",
                        }
                    )
                    break
            if failures:
                break
    return failures


def _actual_outputs_file_refs(data: Any) -> set[str]:
    refs: set[str] = set()

    def add(value: Any) -> None:
        if isinstance(value, str) and value.strip():
            text = value.strip().replace("\\", "/")
            refs.add(text)
            refs.add(text.removeprefix("output/"))
            refs.add(Path(text).name)
        elif isinstance(value, dict):
            for key in ("path", "ref", "file", "filename", "artifact_ref", "package_relative_path"):
                add(value.get(key))

    if not isinstance(data, dict):
        return refs
    for key in (
        "files",
        "files_created",
        "deliverable_refs",
        "artifact_refs",
        "output_refs",
        "generated_files",
        "created_files",
    ):
        for value in _as_list(data.get(key)):
            add(value)
    for key in ("primary_output_ref", "actual_outputs_ref"):
        add(data.get(key))
    return refs


def normalize_contract(contract: dict[str, Any] | None) -> dict[str, Any]:
    contract = dict(contract or {})
    outputs: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for key in ("outputs", "required_outputs", "required_output_files", "files"):
        for item in _as_list(contract.get(key)):
            if isinstance(item, str):
                outputs.append({"path": item})
            elif isinstance(item, dict):
                outputs.append(dict(item))
    for key in ("artifacts", "required_artifacts", "artifact_contracts"):
        for item in _as_list(contract.get(key)):
            if isinstance(item, str):
                artifacts.append({"path": item})
            elif isinstance(item, dict):
                artifacts.append(dict(item))
    if not outputs and not artifacts and _path_value(contract):
        outputs.append(contract)
    return {"outputs": outputs, "artifacts": artifacts}


def verify_output_contract(base: Path, contract: dict[str, Any] | None, *, max_failures: int = 50) -> dict[str, Any]:
    base = base.expanduser()
    normalized = normalize_contract(contract)
    failures: list[dict[str, Any]] = []
    checked: list[dict[str, Any]] = []
    semantic_warnings: list[dict[str, Any]] = []
    required_refs: list[str] = []

    def check_item(item: dict[str, Any], kind: str) -> None:
        rel = _path_value(item)
        if not rel:
            return
        if _hidden_or_forbidden_ref(rel):
            failures.append(
                {
                    "path": rel,
                    "kind": kind,
                    "issue": "forbidden_or_hidden_reference",
                    "repair_hint": "Use visible task inputs and output paths only; do not read expected or hidden reference files.",
                }
            )
            return
        required_refs.append(rel)
        candidates = _candidate_paths(base, rel)
        path = next((candidate for candidate in candidates if candidate.exists() and candidate.is_file()), None)
        fields = _field_names(item)
        semantic_requirements = _semantic_requirements(item)
        entry = {
            "kind": kind,
            "path": rel,
            "exists": bool(path),
            "fields": fields,
            "semantic_requirements": semantic_requirements,
        }
        checked.append(entry)
        for requirement in semantic_requirements:
            semantic_warnings.append(
                {
                    "path": rel,
                    "kind": kind,
                    "requirement": requirement,
                    "status": "requires_semantic_or_grader_check",
                    "repair_hint": "Use task rubric, visible inputs, and grader/verifier feedback to check this requirement; do not infer hidden answers.",
                }
            )
        if not path:
            failures.append({"path": rel, "kind": kind, "issue": "missing_file", "repair_hint": f"Create {rel} in the required output location."})
            return
        if path.stat().st_size == 0:
            failures.append({"path": rel, "kind": kind, "issue": "empty_file", "repair_hint": f"Regenerate {rel}; it is empty."})
            return
        suffix = path.suffix.lower()
        issues: list[str] = []
        fmt = str(item.get("format") or "").lower()
        if suffix == ".json" or fmt == "json":
            issues = _check_json_fields(path, fields)
        elif suffix == ".csv" or fmt == "csv":
            issues = _check_csv_fields(path, fields)
        elif suffix in {".tsv", ".bed"} or fmt in {"tsv", "bed"}:
            issues = _check_csv_fields(path, fields, delimiter="\t") if fields else []
        for issue in issues:
            failures.append({"path": rel, "kind": kind, "issue": issue, "repair_hint": f"Repair {rel} so it matches the required schema."})

    for item in normalized["outputs"]:
        check_item(item, "output")
    for item in normalized["artifacts"]:
        check_item(item, "artifact")

    actual_outputs_candidates = _candidate_paths(base, "output/actual_outputs.json") + _candidate_paths(base, "actual_outputs.json")
    actual_outputs_path = next((candidate for candidate in actual_outputs_candidates if candidate.exists() and candidate.is_file()), None)
    actual_outputs_ok = False
    actual_outputs_file_refs: set[str] = set()
    if actual_outputs_path:
        try:
            data = _read_json(actual_outputs_path)
            actual_outputs_ok = isinstance(data, dict)
            actual_outputs_file_refs = _actual_outputs_file_refs(data)
        except Exception:
            failures.append({"path": "actual_outputs.json", "kind": "output", "issue": "invalid_actual_outputs_json", "repair_hint": "Rewrite actual_outputs.json as valid JSON."})
    else:
        failures.append({"path": "actual_outputs.json", "kind": "output", "issue": "missing_actual_outputs_json", "repair_hint": "Create actual_outputs.json with generated file references."})

    if actual_outputs_ok:
        for rel in required_refs:
            if Path(rel).name == "actual_outputs.json":
                continue
            normalized_refs = {rel, rel.removeprefix("output/"), Path(rel).name}
            if actual_outputs_file_refs and not (normalized_refs & actual_outputs_file_refs):
                failures.append(
                    {
                        "path": rel,
                        "kind": "output",
                        "issue": "actual_outputs_json_missing_required_file_ref",
                        "repair_hint": f"Update actual_outputs.json so it lists {rel}.",
                    }
                )
            elif not actual_outputs_file_refs:
                failures.append(
                    {
                        "path": rel,
                        "kind": "output",
                        "issue": "actual_outputs_json_has_no_file_refs",
                        "repair_hint": "Populate actual_outputs.json files/files_created/deliverable_refs with generated file paths.",
                    }
                )

        actual_outputs_data = _read_json(actual_outputs_path) if actual_outputs_path else {}
        for item in normalized["outputs"]:
            rel = _path_value(item)
            if not rel or Path(rel).name != "actual_outputs.json":
                continue
            for required_file in _as_list(item.get("actual_outputs_required_files")):
                if isinstance(required_file, str):
                    normalized_required = {required_file, required_file.removeprefix("output/"), Path(required_file).name}
                    if not (normalized_required & actual_outputs_file_refs):
                        failures.append(
                            {
                                "path": rel,
                                "kind": "output",
                                "issue": "actual_outputs_json_missing_declared_file",
                                "repair_hint": f"Update actual_outputs.json so it lists {required_file}.",
                            }
                        )
            for field, expected_item_type in _list_item_type_map(item).items():
                exists, value = _lookup_path(actual_outputs_data, field)
                if not exists or not isinstance(value, list):
                    continue
                for index, entry in enumerate(value):
                    if not _type_matches(entry, expected_item_type):
                        failures.append(
                            {
                                "path": rel,
                                "kind": "output",
                                "issue": f"actual_outputs_json_list_item_type_mismatch:{field}.{index}",
                                "repair_hint": f"Rewrite {field} entries in actual_outputs.json as {expected_item_type} values.",
                            }
                        )
                        break

    failures = failures[:max_failures]
    return {
        "contract_verification_passed": not failures,
        "actual_outputs_json_ok": actual_outputs_ok,
        "checked": checked,
        "failures": failures,
        "failure_count": len(failures),
        "semantic_warnings": semantic_warnings[:max_failures],
        "semantic_warning_count": len(semantic_warnings),
    }


def semantic_verify_outputs(
    base: Path,
    contract: dict[str, Any] | None,
    *,
    source_contract: dict[str, Any] | None = None,
    max_failures: int = 50,
) -> dict[str, Any]:
    """Visible-evidence semantic checks for scoring-critical output contracts.

    This intentionally does not read expected/ or hidden reference files. It checks
    declared schema/type/manifest invariants and reports uncertain scorer-critical
    checks as targeted warnings rather than pretending to know hidden answers.
    """
    base = base.expanduser()
    structural = verify_output_contract(base, contract, max_failures=max_failures)
    normalized = normalize_contract(contract)
    source_normalized = normalize_contract(source_contract)
    failures: list[dict[str, Any]] = list(structural.get("failures") or [])
    warnings: list[dict[str, Any]] = list(structural.get("semantic_warnings") or [])
    visible_checks_used = [
        "required_file_existence",
        "json_parse",
        "csv_tsv_header_check",
        "actual_outputs_file_refs",
        "declared_numeric_fields",
        "declared_non_empty_fields",
        "declared_row_count_bounds",
        "target_source_contract_conflict",
    ]
    checked_paths: set[str] = set()

    def add_failure(rel: str, issue: str, repair_hint: str, *, kind: str = "output") -> None:
        failures.append({"path": rel, "kind": kind, "issue": issue, "repair_hint": repair_hint})

    def check_item(item: dict[str, Any], kind: str) -> None:
        rel = _path_value(item)
        if not rel or _hidden_or_forbidden_ref(rel):
            return
        checked_paths.add(rel)
        path = next((candidate for candidate in _candidate_paths(base, rel) if candidate.exists() and candidate.is_file()), None)
        if not path:
            return
        fmt = str(item.get("format") or path.suffix.lstrip(".")).lower()
        numeric_fields = _numeric_fields(item)
        non_empty_fields = _non_empty_fields(item)
        expected_types = _expected_type_map(item)
        list_item_types = _list_item_type_map(item)
        min_rows = item.get("min_rows")
        max_rows = item.get("max_rows")
        exact_rows = item.get("row_count") or item.get("expected_row_count")
        min_size = int(item.get("min_bytes") or 1)
        if path.stat().st_size < min_size:
            add_failure(rel, "artifact_too_small_or_empty", f"Regenerate {rel}; it is too small to satisfy the declared artifact contract.", kind=kind)
            return
        if fmt == "json" or path.suffix.lower() == ".json":
            data = _json_data_at_path(path)
            if data is None:
                return
            rows = data if isinstance(data, list) else [data]
            if exact_rows is not None and isinstance(data, list) and len(data) != int(exact_rows):
                add_failure(rel, "json_row_count_mismatch", f"Adjust {rel} so it has exactly {exact_rows} rows/items.", kind=kind)
            if min_rows is not None and isinstance(data, list) and len(data) < int(min_rows):
                add_failure(rel, "json_row_count_below_minimum", f"Add the missing rows/items to {rel}; expected at least {min_rows}.", kind=kind)
            if max_rows is not None and isinstance(data, list) and len(data) > int(max_rows):
                add_failure(rel, "json_row_count_above_maximum", f"Reduce {rel} to at most {max_rows} rows/items.", kind=kind)
            for row in rows:
                if not isinstance(row, dict):
                    continue
                for field in non_empty_fields:
                    exists, value = _lookup_path(row, field)
                    if not exists or value in (None, "", [], {}):
                        add_failure(rel, f"json_field_empty:{field}", f"Populate JSON field {field} in {rel}.", kind=kind)
                for field in numeric_fields:
                    exists, value = _lookup_path(row, field)
                    if not exists or _coerce_number(value) is None:
                        add_failure(rel, f"json_field_not_numeric:{field}", f"Write numeric values for JSON field {field} in {rel}.", kind=kind)
                for field, expected_type in expected_types.items():
                    exists, value = _lookup_path(row, field)
                    if not exists or not _type_matches(value, expected_type):
                        add_failure(rel, f"json_field_type_mismatch:{field}", f"Write {field} in {rel} as {expected_type}.", kind=kind)
                for field, expected_item_type in list_item_types.items():
                    exists, value = _lookup_path(row, field)
                    if not exists or not isinstance(value, list):
                        continue
                    for index, entry in enumerate(value):
                        if not _type_matches(entry, expected_item_type):
                            add_failure(rel, f"json_list_item_type_mismatch:{field}.{index}", f"Write {field} list entries in {rel} as {expected_item_type} values.", kind=kind)
                            break
        elif fmt in {"csv", "tsv", "bed"} or path.suffix.lower() in {".csv", ".tsv", ".bed"}:
            delimiter = "\t" if fmt in {"tsv", "bed"} or path.suffix.lower() in {".tsv", ".bed"} else ","
            try:
                _header, rows = _csv_rows(path, delimiter=delimiter)
            except Exception:
                return
            if exact_rows is not None and len(rows) != int(exact_rows):
                add_failure(rel, "csv_row_count_mismatch", f"Adjust {rel} so it has exactly {exact_rows} data rows.", kind=kind)
            if min_rows is not None and len(rows) < int(min_rows):
                add_failure(rel, "csv_row_count_below_minimum", f"Add the missing rows to {rel}; expected at least {min_rows}.", kind=kind)
            if max_rows is not None and len(rows) > int(max_rows):
                add_failure(rel, "csv_row_count_above_maximum", f"Reduce {rel} to at most {max_rows} data rows.", kind=kind)
            if not rows and item.get("scoring_critical", True):
                add_failure(rel, "csv_has_no_data_rows", f"Populate {rel} with data rows, not just a header.", kind=kind)
            for row in rows:
                for field in non_empty_fields:
                    if row.get(field) in (None, ""):
                        add_failure(rel, f"csv_field_empty:{field}", f"Populate CSV column {field} in {rel}.", kind=kind)
                for field in numeric_fields:
                    if _coerce_number(row.get(field)) is None:
                        add_failure(rel, f"csv_field_not_numeric:{field}", f"Write numeric values for CSV column {field} in {rel}.", kind=kind)
                for field, expected_type in expected_types.items():
                    value = row.get(field)
                    if expected_type in {"number", "numeric", "float", "int", "integer"} and _coerce_number(value) is None:
                        add_failure(rel, f"csv_field_type_mismatch:{field}", f"Write {field} in {rel} as {expected_type}.", kind=kind)
            for failure in _derived_value_failures(base, item, rel, rows):
                failures.append(failure)

    for item in normalized["outputs"]:
        check_item(item, "output")
    for item in normalized["artifacts"]:
        check_item(item, "artifact")

    if source_normalized["outputs"] and normalized["outputs"]:
        target_paths = {_path_value(item) for item in normalized["outputs"] if _path_value(item)}
        source_paths = {_path_value(item) for item in source_normalized["outputs"] if _path_value(item)}
        source_only = sorted(path for path in source_paths - target_paths if path and not _hidden_or_forbidden_ref(path))
        for path in source_only[:10]:
            warnings.append(
                {
                    "path": path,
                    "kind": "source_contract",
                    "requirement": "source_contract_path_not_in_target_contract",
                    "status": "do_not_copy_source_output_contract_blindly",
                    "repair_hint": "Follow the target mirror contract; do not add source-only output paths unless the target task asks for them.",
                }
            )

    unique_failures: list[dict[str, Any]] = []
    seen = set()
    for failure in failures:
        key = (failure.get("path"), failure.get("issue"))
        if key not in seen:
            seen.add(key)
            unique_failures.append(failure)
    unique_failures = unique_failures[:max_failures]
    status = "fail" if unique_failures else "uncertain" if warnings else "pass"
    repair_allowed = bool(unique_failures) and status == "fail" and len({f.get("path") for f in unique_failures}) <= 3
    return {
        "semantic_verification_status": status,
        "scorer_critical_failures": unique_failures,
        "non_blocking_warnings": warnings[:max_failures],
        "evidence_refs": sorted(checked_paths),
        "visible_checks_used": visible_checks_used,
        "hidden_reference_used": False,
        "repair_allowed": repair_allowed,
        "repair_scope": sorted({str(f.get("path")) for f in unique_failures if f.get("path")}),
        "do_not_modify": sorted(checked_paths - {str(f.get("path")) for f in unique_failures if f.get("path")}),
        "reasoning_summary": "Semantic verification used visible contract, produced artifacts, and actual_outputs refs only.",
        "structural_contract_verification": structural,
    }


def build_repair_plan(verification: dict[str, Any], *, repair_id: str | None = None) -> dict[str, Any]:
    failures = verification.get("failures") or []
    semantic_failures = verification.get("scorer_critical_failures") or []
    all_failures = list(failures) + list(semantic_failures)
    repair_allowed = verification.get("repair_allowed")
    if repair_allowed is None:
        repair_allowed = bool(all_failures)
    files_to_modify = sorted({str(f.get("path")) for f in all_failures if f.get("path")})
    do_not_modify = sorted(set(verification.get("do_not_modify") or []))
    risk = "high" if len(files_to_modify) > 3 or verification.get("semantic_verification_status") == "uncertain" else "medium" if len(files_to_modify) > 1 else "low"
    return {
        "repair_id": repair_id or "contract_repair_001",
        "repair_required": bool(all_failures),
        "repair_allowed": bool(repair_allowed),
        "failed_items": all_failures,
        "issues_to_fix": all_failures,
        "files_to_modify": files_to_modify,
        "files_to_preserve": do_not_modify,
        "specific_changes": [failure.get("repair_hint") for failure in all_failures if failure.get("repair_hint")],
        "verification_after_repair": ["run structural contract verifier", "run semantic verifier", "run local grader if available"],
        "risk_of_regression": risk,
        "broad_rewrite_allowed": False,
        "preserve_passing_items": True,
        "repair_scope": "target_failed_contract_items_only",
        "repair_steps": [
            "Read only the listed contract failures and identify the affected files.",
            "Preserve files that already passed contract checks.",
            "Regenerate or patch each missing or malformed required output from current visible inputs.",
            "Reopen CSV/JSON/TSV files and check required columns or keys.",
            "Update actual_outputs.json so it lists generated files exactly.",
            "Run the contract verifier again before finishing.",
        ] if all_failures and repair_allowed else [],
    }


def build_repair_result(
    repair_plan: dict[str, Any],
    *,
    before_verification: dict[str, Any] | None = None,
    after_verification: dict[str, Any] | None = None,
    before_score: float | None = None,
    after_score: float | None = None,
    files_modified: list[str] | None = None,
    reverted: bool = False,
    notes: str = "",
) -> dict[str, Any]:
    before_status = (before_verification or {}).get("semantic_verification_status") or (before_verification or {}).get("contract_verification_passed")
    after_status = (after_verification or {}).get("semantic_verification_status") or (after_verification or {}).get("contract_verification_passed")
    regression = False
    if before_score is not None and after_score is not None and after_score < before_score:
        regression = True
    if before_status in {"pass", True} and after_status not in {"pass", True}:
        regression = True
    return {
        "repair_id": repair_plan.get("repair_id") or "contract_repair_001",
        "files_modified": files_modified or [],
        "files_preserved": repair_plan.get("files_to_preserve") or [],
        "before_score": before_score,
        "after_score": after_score,
        "before_contract_status": before_status,
        "after_contract_status": after_status,
        "regression_detected": regression,
        "reverted": bool(reverted),
        "notes": notes,
    }
