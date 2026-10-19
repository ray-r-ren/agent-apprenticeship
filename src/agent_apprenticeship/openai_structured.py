from __future__ import annotations
import importlib, json, os, re, signal, sys, threading, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path
from typing import Any
from pydantic import BaseModel
from .config import get_settings
from .io import write_json
from .env import redact_secrets, contains_secret
from .recipes import MODEL_PROVIDER_RECIPES
from .role_runners import RoleResult
from .llm_output_normalizer import normalize_role_output

class JsonExtractionError(ValueError):
    pass


def _bool(v: bool) -> str:
    return str(bool(v)).lower()

def _safe_error_message(exc: BaseException | str | None) -> str | None:
    if exc is None:
        return None
    return redact_secrets(str(exc))[:1000]

def _repo_src_path() -> str:
    return str(Path(__file__).resolve().parents[1])

def _drop_local_pydantic_if_needed() -> dict[str, Any]:
    """Avoid src/pydantic shim shadowing real pydantic for the OpenAI SDK.

    Deterministic tests in this kata run without third-party deps, so the repo has a tiny
    local pydantic shim. Real OpenAI SDK imports require real pydantic. When available,
    temporarily remove repo src from import lookup and evict the shim before importing
    `openai`.
    """
    src = _repo_src_path()
    # In an installed package, parents[1] is usually site-packages. Removing
    # that path prevents the OpenAI SDK itself from being imported. Only strip
    # the path when it looks like the old repo-local pydantic shim directory:
    # it has pydantic but not the OpenAI SDK alongside it.
    if not (Path(src) / "pydantic" / "__init__.py").exists() or (Path(src) / "openai").exists():
        return {'removed_paths': [], 'removed_modules': {}}
    removed_paths = []
    for p in list(sys.path):
        if Path(p or '.').resolve().as_posix() == Path(src).resolve().as_posix():
            sys.path.remove(p); removed_paths.append(p)
    removed_modules = {}
    for name, mod in list(sys.modules.items()):
        if name == 'pydantic' or name.startswith('pydantic.'):
            file = getattr(mod, '__file__', '') or ''
            if '/src/pydantic/' in file or file.endswith('/src/pydantic/__init__.py'):
                removed_modules[name] = mod
                del sys.modules[name]
    return {'removed_paths': removed_paths, 'removed_modules': removed_modules}

def _restore_import_state(state: dict[str, Any]) -> None:
    for p in reversed(state.get('removed_paths', [])):
        if p not in sys.path:
            sys.path.insert(0, p)
    # Do not restore local pydantic if real pydantic was imported successfully.
    if 'pydantic' not in sys.modules:
        sys.modules.update(state.get('removed_modules', {}))

def import_openai_sdk():
    if 'openai' in sys.modules:
        return sys.modules['openai']
    state = _drop_local_pydantic_if_needed()
    try:
        return importlib.import_module('openai')
    finally:
        _restore_import_state(state)

def get_openai_status() -> dict[str, Any]:
    settings = get_settings()
    status = {
        'openai_sdk_import_ok': False,
        'openai_api_key_visible': bool(settings.openai_api_key),
        'openai_client_constructed_ok': False,
        'openai_available': False,
        'error_type': None,
        'error_message': None,
    }
    try:
        mod = import_openai_sdk()
        status['openai_sdk_import_ok'] = True
    except Exception as exc:
        status['error_type'] = type(exc).__name__
        status['error_message'] = _safe_error_message(exc)
        return status
    if not settings.openai_api_key:
        status['error_type'] = 'OpenAIKeyMissing'
        status['error_message'] = 'OPENAI_API_KEY is not visible.'
        return status
    # Availability means SDK import + visible key. Client construction is reported separately.
    status['openai_available'] = True
    try:
        mod.OpenAI(api_key=settings.openai_api_key)
        status['openai_client_constructed_ok'] = True
    except Exception as exc:
        status['error_type'] = type(exc).__name__
        status['error_message'] = _safe_error_message(exc)
    return status

def openai_available() -> bool:
    return bool(get_openai_status().get('openai_available'))

OPENAI_COMPATIBLE_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
}

def _configured_provider(provider_id: str | None = None) -> str:
    settings = get_settings()
    return provider_id or settings.model_provider or "openai"

def _provider_key_env(provider_id: str, settings=None) -> str | None:
    settings = settings or get_settings()
    recipe = MODEL_PROVIDER_RECIPES.get(provider_id)
    if settings.model_provider == provider_id and settings.model_provider_api_key_env:
        return settings.model_provider_api_key_env
    return recipe.api_key_env_var if recipe else None

def _provider_key(provider_id: str, settings=None) -> tuple[str | None, str | None]:
    settings = settings or get_settings()
    env_var = _provider_key_env(provider_id, settings)
    if provider_id == "openai":
        return env_var or "OPENAI_API_KEY", settings.openai_api_key or os.getenv(env_var or "OPENAI_API_KEY")
    value = os.getenv(env_var or "") if env_var else None
    if provider_id == "google" and not value:
        fallback = "GOOGLE_API_KEY"
        value = os.getenv(fallback)
        if value:
            env_var = fallback
    return env_var, value

def _provider_model(provider_id: str, model_override: str | None = None) -> str:
    settings = get_settings()
    recipe = MODEL_PROVIDER_RECIPES.get(provider_id)
    if model_override:
        return model_override
    if settings.model_provider == provider_id and settings.model_provider_model:
        return settings.model_provider_model
    if provider_id == "openai":
        return settings.openai_model
    return (recipe.default_model if recipe else None) or settings.openai_model

def _provider_max_output_tokens() -> int:
    try:
        value = int(os.getenv("AA_MODEL_SMOKE_MAX_TOKENS") or "512")
    except ValueError:
        value = 2048
    return max(256, min(value, 4096))


def _provider_request_timeout_seconds() -> int:
    settings = get_settings()
    raw = os.getenv("AA_ROLE_TIMEOUT_SECONDS") or os.getenv("AA_MODEL_ROLE_TIMEOUT_SECONDS")
    if raw:
        try:
            return max(10, min(int(raw), settings.task_timeout_seconds))
        except ValueError:
            pass
    return max(10, min(30, settings.task_timeout_seconds))


def _provider_request_retry_count() -> int:
    raw = os.getenv("AA_ROLE_RETRY_COUNT") or os.getenv("AA_MODEL_ROLE_RETRY_COUNT")
    if raw:
        try:
            return max(1, min(int(raw), 3))
        except ValueError:
            pass
    return 1


def _provider_wall_timeout_seconds() -> int:
    settings = get_settings()
    raw = os.getenv("AA_ROLE_WALL_TIMEOUT_SECONDS") or os.getenv("AA_MODEL_ROLE_WALL_TIMEOUT_SECONDS")
    if raw:
        try:
            return max(1, min(int(raw), settings.task_timeout_seconds))
        except ValueError:
            pass
    return max(12, min(_provider_request_timeout_seconds() + 2, settings.task_timeout_seconds))


def _with_wall_timeout(label: str, fn):
    timeout = _provider_wall_timeout_seconds()
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "setitimer"):
        return fn()
    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.setitimer(signal.ITIMER_REAL, 0)

    def _raise_timeout(_signum, _frame):
        raise TimeoutError(f"{label} timed out after {timeout} seconds.")

    try:
        signal.signal(signal.SIGALRM, _raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, timeout)
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer and old_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, old_timer[0], old_timer[1])


def _construct_openai_client(mod: Any, kwargs: dict[str, Any]) -> Any:
    try:
        return mod.OpenAI(**kwargs)
    except TypeError as exc:
        if "timeout" not in kwargs or "timeout" not in str(exc):
            raise
        return mod.OpenAI(**{k: v for k, v in kwargs.items() if k != "timeout"})


def get_model_provider_status(provider_id: str | None = None) -> dict[str, Any]:
    provider_id = _configured_provider(provider_id)
    settings = get_settings()
    recipe = MODEL_PROVIDER_RECIPES.get(provider_id)
    env_var, key = _provider_key(provider_id, settings)
    status = {
        "provider_id": provider_id,
        "provider_display": recipe.display_name if recipe else provider_id,
        "model": _provider_model(provider_id),
        "api_key_env_var": env_var,
        "api_key_visible": bool(key),
        "adapter_available": provider_id in MODEL_PROVIDER_RECIPES,
        "provider_available": False,
        "client_constructed_ok": False,
        "error_type": None,
        "error_message": None,
    }
    if not recipe:
        status["error_type"] = "UnsupportedProvider"
        status["error_message"] = f"Unsupported Mentor Model Provider: {provider_id}"
        return status
    if not key:
        status["error_type"] = "APIKeyMissing"
        status["error_message"] = f"{env_var or recipe.api_key_env_var} is not visible."
        return status
    if provider_id in {"openai", "openrouter"}:
        try:
            mod = import_openai_sdk()
            status["sdk_import_ok"] = True
            kwargs = {"api_key": key}
            if provider_id in OPENAI_COMPATIBLE_BASE_URLS:
                kwargs["base_url"] = OPENAI_COMPATIBLE_BASE_URLS[provider_id]
            _construct_openai_client(mod, kwargs)
            status["client_constructed_ok"] = True
            status["provider_available"] = True
        except Exception as exc:
            status["error_type"] = type(exc).__name__
            status["error_message"] = _safe_error_message(exc)
        return status
    status["client_constructed_ok"] = True
    status["provider_available"] = True
    return status

def classify_model_provider_error(exc: BaseException | str | None, status_code: int | None = None) -> tuple[str, str]:
    text = redact_secrets(str(exc or ""))[:1000]
    low = text.lower()
    if status_code in {401, 403} or any(n in low for n in ["unauthorized", "forbidden", "invalid api key", "authentication", "permission denied"]):
        return "auth_error", text or "Mentor Model Provider authentication failed."
    if "insufficient balance" in low:
        return "insufficient_balance", text or "Mentor Model Provider account balance is insufficient."
    if status_code in {402, 429} or any(n in low for n in ["quota", "credit", "billing", "rate limit", "insufficient"]):
        return "quota_or_credit_error", text or "Mentor Model Provider quota or credit limit reached."
    if any(n in low for n in ["timeout", "timed out"]):
        return "timeout", text or "Mentor Model Provider request timed out."
    if status_code == 404 or any(n in low for n in ["model_not_found", "model not found", "unknown model", "invalid model"]):
        return "bad_model", text or "Mentor Model Provider model was not found."
    if any(n in low for n in ["network", "name or service not known", "temporary failure", "connection refused", "connection reset"]):
        return "network_error", text or "Mentor Model Provider network error."
    return "provider_error", text or "Mentor Model Provider request failed."

def _http_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        kind, msg = classify_model_provider_error(body or exc, exc.code)
        raise RuntimeError(f"{kind}: {msg}") from exc
    except TimeoutError as exc:
        kind, msg = classify_model_provider_error(exc)
        raise RuntimeError(f"{kind}: {msg}") from exc

def _openai_compatible_response(provider_id: str, model_name: str, prompt: str) -> str:
    settings = get_settings()
    _env_var, key = _provider_key(provider_id, settings)
    mod = import_openai_sdk()
    kwargs = {"api_key": key, "timeout": _provider_request_timeout_seconds()}
    if provider_id in OPENAI_COMPATIBLE_BASE_URLS:
        kwargs["base_url"] = OPENAI_COMPATIBLE_BASE_URLS[provider_id]
    if provider_id == "openrouter":
        try:
            kwargs["default_headers"] = {
                "HTTP-Referer": "https://github.com/Forsy-AI/agent-apprenticeship",
                "X-Title": "Agent Apprenticeship",
            }
        except Exception:
            pass
    client = _construct_openai_client(mod, kwargs)
    if provider_id == "openai":
        return _response_text(_responses_create_with_retry(client, model_name, prompt))
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=_provider_max_output_tokens(),
    )
    choice = response.choices[0]
    return str(choice.message.content or "")

def _anthropic_response(model_name: str, prompt: str) -> str:
    settings = get_settings()
    _env_var, key = _provider_key("anthropic", settings)
    payload = {
        "model": model_name,
        "max_tokens": _provider_max_output_tokens(),
        "messages": [{"role": "user", "content": prompt}],
    }
    data = _http_json(
        "https://api.anthropic.com/v1/messages",
        {"x-api-key": key or "", "anthropic-version": "2023-06-01"},
        payload,
        timeout=_provider_request_timeout_seconds(),
    )
    parts = data.get("content") or []
    text = "".join(str(part.get("text") or "") for part in parts if isinstance(part, dict))
    if not text:
        raise RuntimeError("provider_response_format: Anthropic response did not contain text content.")
    return text

def _google_response(model_name: str, prompt: str) -> str:
    settings = get_settings()
    _env_var, key = _provider_key("google", settings)
    query = urllib.parse.urlencode({"key": key or ""})
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{urllib.parse.quote(model_name, safe='')}:generateContent?{query}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "maxOutputTokens": _provider_max_output_tokens()},
    }
    data = _http_json(url, {}, payload, timeout=_provider_request_timeout_seconds())
    try:
        return str(data["candidates"][0]["content"]["parts"][0]["text"])
    except Exception as exc:
        raise RuntimeError("provider_response_format: Google response did not contain candidate text.") from exc

def _provider_completion_text(provider_id: str, model_name: str, prompt: str) -> str:
    if provider_id in {"openai", "openrouter"}:
        return _openai_compatible_response(provider_id, model_name, prompt)
    if provider_id == "anthropic":
        return _anthropic_response(model_name, prompt)
    if provider_id == "google":
        return _google_response(model_name, prompt)
    raise RuntimeError(f"Unsupported Mentor Model Provider: {provider_id}")

def _provider_completion_text_with_retry(provider_id: str, model_name: str, prompt: str) -> str:
    last: BaseException | None = None
    attempts = _provider_request_retry_count()
    for i in range(attempts):
        try:
            return _provider_completion_text(provider_id, model_name, prompt)
        except Exception as exc:
            last = exc
            kind, _msg = classify_model_provider_error(exc)
            if i >= attempts - 1 or kind in {"auth_error", "quota_or_credit_error", "bad_model"}:
                raise
            time.sleep(0.75)
    raise last or RuntimeError("Mentor Model Provider request failed.")

def _json_repair_candidates(cand: str) -> list[tuple[str, str]]:
    repaired=[]
    try:
        from json_repair import repair_json  # type: ignore
        repaired_text=repair_json(cand)
        if repaired_text and repaired_text != cand:
            repaired.append((repaired_text, 'json_repair'))
    except Exception:
        pass
    local=re.sub(r',\s*([}\]])', r'\1', cand)
    if local != cand:
        repaired.append((local, 'removed_trailing_commas'))
    return repaired

def extract_json_object(text: str, return_metadata: bool=False):
    if not text or not text.strip():
        raise JsonExtractionError('No text to parse as JSON.')
    candidates: list[tuple[str, str]] = [(text.strip(), 'direct')]
    for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE):
        candidates.append((match.group(1).strip(), 'markdown_fence'))
    for start, ch in enumerate(text):
        if ch not in '{[':
            continue
        stack=[]; in_str=False; esc=False
        for idx in range(start, len(text)):
            c=text[idx]
            if in_str:
                if esc: esc=False
                elif c == '\\': esc=True
                elif c == '"': in_str=False
                continue
            if c == '"': in_str=True
            elif c in '{[': stack.append(c)
            elif c in '}]':
                if not stack: break
                opening=stack.pop()
                if (opening,c) not in [('{','}'),('[',']')]: break
                if not stack:
                    candidates.append((text[start:idx+1], 'balanced_brace_extraction')); break
    expanded=[]
    for cand, reason in candidates:
        expanded.append((cand, reason, False))
        for repaired, repair_reason in _json_repair_candidates(cand):
            expanded.append((repaired, f'{reason}+{repair_reason}', True))
    errors=[]
    for cand, reason, repaired in expanded:
        try:
            obj=json.loads(cand)
            if isinstance(obj, dict):
                meta={'repaired_json': repaired, 'repair_reason': reason if repaired else None, 'json_extraction_reason': reason}
                return (obj, meta) if return_metadata else obj
            raise JsonExtractionError('Parsed JSON was not an object.')
        except Exception as exc:
            errors.append(type(exc).__name__)
    raise JsonExtractionError('Could not extract a valid JSON object from model output: ' + ','.join(errors[:5]))

def _validate_role_output(
    role: str,
    raw: str,
    output_model: type[BaseModel],
    normalizer_context: dict[str, Any] | None,
    out_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    parsed,json_meta=extract_json_object(raw, return_metadata=True)
    write_json(out_dir/'raw_parsed_output.json', parsed)
    initial_validation_error=None
    try:
        output_model.model_validate(parsed)
    except Exception as first_exc:
        initial_validation_error=_safe_error_message(first_exc)
    normalized=normalize_role_output(role, parsed, normalizer_context)
    normalized_used=normalized != parsed or initial_validation_error is not None
    output_model.model_validate(normalized)
    if normalized_used:
        write_json(out_dir/'normalization_report.json', {'role': role, 'normalization_applied': True, 'initial_validation_error': initial_validation_error, 'raw_keys': list(parsed.keys()), 'normalized_keys': list(normalized.keys()) if isinstance(normalized, dict) else [], 'extras_preserved_in_metadata': True})
    return normalized, json_meta, normalized_used

def _strict_json_retry_prompt(role: str, original_prompt: str, error_message: str | None) -> str:
    return (
        "Your previous response could not be parsed or validated. "
        "Return ONLY one compact valid JSON object, with double-quoted JSON keys and no markdown/prose. "
        f"Role: {role}. Error to fix: {error_message or 'invalid JSON'}.\n\n"
        f"{original_prompt}"
    )

def _transient_provider_error(exc: BaseException) -> bool:
    name=type(exc).__name__.lower(); msg=str(exc).lower()
    needles=['timeout','rate','temporar','connection reset','disconnect','websocket','service unavailable','gateway','overloaded']
    return any(n in name or n in msg for n in needles)

def _responses_create_with_retry(client: Any, model_name: str, prompt: str, attempts: int = 1):
    last=None
    for i in range(attempts):
        try:
            return client.responses.create(model=model_name, input=prompt)
        except Exception as exc:
            last=exc
            if i >= attempts-1 or not _transient_provider_error(exc):
                raise
            time.sleep(min(4.0, 0.5 * (2 ** i)))
    raise last  # type: ignore[misc]

def _response_text(response: Any) -> str:
    text = getattr(response, 'output_text', None)
    if text is not None:
        return str(text)
    if isinstance(response, dict):
        return str(response.get('output_text') or response.get('text') or json.dumps(response))
    try:
        return response.model_dump_json()
    except Exception:
        return str(response)

def run_structured_role(role: str, prompt: str, output_model: type[BaseModel], out_dir: Path, allow_fallback=False, require_validation=False, model_override: str | None=None, normalizer_context: dict[str, Any] | None=None, provider_override: str | None=None) -> RoleResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/'prompt.md').write_text(redact_secrets(prompt))
    settings=get_settings()
    provider_id=_configured_provider(provider_override)
    model_name=_provider_model(provider_id, model_override)
    start=time.time()
    status=get_model_provider_status(provider_id)
    meta={
        'mentor_model_provider': provider_id,
        'mentor_model_provider_display': status.get('provider_display'),
        'mentor_model_provider_api_key_visible': bool(status.get('api_key_visible')),
        'mentor_model_provider_adapter_available': bool(status.get('adapter_available')),
        'mentor_model_provider_client_constructed_ok': bool(status.get('client_constructed_ok')),
        'mentor_model_provider_available': bool(status.get('provider_available')),
    }
    if provider_id == 'openai':
        openai_status=get_openai_status()
        meta.update({k: openai_status.get(k) for k in ['openai_sdk_import_ok','openai_api_key_visible','openai_client_constructed_ok','openai_available']})
    if not status['provider_available']:
        result=RoleResult(role=role, provider=provider_id, model=model_name, live_call_ok=False, structured_output_validation_ok=False, prompt_ref='prompt.md', output_ref='raw_output.txt', parsed_output_ref='parsed_output.json', error_type=status.get('error_type') or 'MentorModelProviderUnavailable', error_message=status.get('error_message') or 'Mentor Model Provider adapter or API key unavailable; no live call made.', duration_seconds=time.time()-start, metadata_json={**meta, 'allow_fallback': allow_fallback})
        (out_dir/'raw_output.txt').write_text(result.error_message or '')
        write_json(out_dir/'parsed_output.json', {'error': result.error_message})
        write_json(out_dir/'role_result.json', result)
        if not allow_fallback:
            raise RuntimeError(result.error_message)
        return result
    try:
        raw=_with_wall_timeout(
            f"{status.get('provider_display') or provider_id} {role}",
            lambda: _provider_completion_text_with_retry(provider_id, model_name, prompt),
        )
        (out_dir/'raw_output.txt').write_text(redact_secrets(raw))
        try:
            normalized,json_meta,normalized_used=_validate_role_output(role, raw, output_model, normalizer_context, out_dir)
            write_json(out_dir/'parsed_output.json', normalized)
            result=RoleResult(role=role, provider=provider_id, model=model_name, live_call_ok=True, structured_output_validation_ok=True, prompt_ref='prompt.md', output_ref='raw_output.txt', parsed_output_ref='parsed_output.json', error_type=None, error_message=None, duration_seconds=time.time()-start, metadata_json={**meta, **json_meta, 'mentor_model_provider_live_call_ok': True, f'{provider_id}_live_call_ok': True, 'normalization_applied': normalized_used})
        except Exception as exc:
            first_error=_safe_error_message(exc)
            retry_raw=None
            try:
                retry_prompt=_strict_json_retry_prompt(role, prompt, first_error)
                retry_raw=_with_wall_timeout(
                    f"{status.get('provider_display') or provider_id} {role} structured retry",
                    lambda: _provider_completion_text_with_retry(provider_id, model_name, retry_prompt),
                )
                (out_dir/'raw_output.retry.txt').write_text(redact_secrets(retry_raw))
                normalized,json_meta,normalized_used=_validate_role_output(role, retry_raw, output_model, normalizer_context, out_dir)
                write_json(out_dir/'parsed_output.json', normalized)
                result=RoleResult(role=role, provider=provider_id, model=model_name, live_call_ok=True, structured_output_validation_ok=True, prompt_ref='prompt.md', output_ref='raw_output.retry.txt', parsed_output_ref='parsed_output.json', error_type=None, error_message=None, duration_seconds=time.time()-start, metadata_json={**meta, **json_meta, 'mentor_model_provider_live_call_ok': True, f'{provider_id}_live_call_ok': True, 'normalization_applied': normalized_used, 'structured_retry_used': True, 'initial_parse_or_validation_error': first_error})
            except Exception as retry_exc:
                write_json(out_dir/'parsed_output.json', {'parse_or_validation_error': _safe_error_message(retry_exc), 'initial_parse_or_validation_error': first_error, 'raw_output_preserved_ref': 'raw_output.txt', 'retry_output_preserved_ref': 'raw_output.retry.txt' if retry_raw is not None else None, 'raw_parsed_output_ref': 'raw_parsed_output.json' if (out_dir/'raw_parsed_output.json').exists() else None})
                result=RoleResult(role=role, provider=provider_id, model=model_name, live_call_ok=True, structured_output_validation_ok=False, prompt_ref='prompt.md', output_ref='raw_output.txt', parsed_output_ref='parsed_output.json', error_type=type(retry_exc).__name__, error_message=_safe_error_message(retry_exc), duration_seconds=time.time()-start, metadata_json={**meta, 'mentor_model_provider_live_call_ok': True, f'{provider_id}_live_call_ok': True, 'fallback_used': allow_fallback, 'structured_retry_used': retry_raw is not None, 'initial_parse_or_validation_error': first_error})
                if require_validation:
                    write_json(out_dir/'role_result.json', result)
                    raise RuntimeError(result.error_message)
    except Exception as exc:
        err_type, err_msg = classify_model_provider_error(exc)
        (out_dir/'raw_output.txt').write_text(_safe_error_message(exc) or '')
        write_json(out_dir/'parsed_output.json', {'error': err_msg})
        result=RoleResult(role=role, provider=provider_id, model=model_name, live_call_ok=False, structured_output_validation_ok=False, prompt_ref='prompt.md', output_ref='raw_output.txt', parsed_output_ref='parsed_output.json', error_type=err_type, error_message=err_msg, duration_seconds=time.time()-start, metadata_json={**meta, 'mentor_model_provider_live_call_ok': False, f'{provider_id}_live_call_ok': False, 'fallback_used': allow_fallback})
        if require_validation and not allow_fallback:
            write_json(out_dir/'role_result.json', result)
            raise
    write_json(out_dir/'role_result.json', result)
    return result

def _json_smoke_prompt(label: str, skeleton: dict[str, Any]) -> str:
    return (
        f"Return ONLY this compact valid JSON object for {label}. "
        "Keep every key and JSON type. Use double quotes. No markdown, no prose.\n"
        + json.dumps(skeleton, separators=(",", ":"))
    )

# Compact schema prompts for llm-smoke. They use literal JSON skeletons so
# providers do not need to infer local Pydantic class definitions.
def smoke_prompts() -> dict[str, tuple[str, type[BaseModel]]]:
    from .schemas import TaskIntakeSpec, TaskIntakeQualityReport, RubricSpec, RubricQualityReport, GraderResult, VerifierResult, EvaluatorFeedback, RevisionPlan

    class _IntakeSmokeOutput(BaseModel):
        task_intake_spec: TaskIntakeSpec
        task_intake_quality_report: TaskIntakeQualityReport

    class _RubricSmokeOutput(BaseModel):
        rubric_spec: RubricSpec
        rubric_quality_report: RubricQualityReport

    class _EvaluatorSmokeOutput(BaseModel):
        evaluator_feedback: EvaluatorFeedback
        revision_plan: RevisionPlan

    rubric_item = {
        "rubric_item_id": "ri_1",
        "criterion_name": "artifact",
        "criterion_description": "hello.txt exists and contains hello",
        "weight": 1.0,
        "score_min": 0.0,
        "score_max": 1.0,
        "pass_threshold": 0.7,
        "observable_evidence": ["artifacts/hello.txt"],
        "required_artifacts": ["hello.txt"],
        "scoring_method": "hybrid",
        "worker_visible": True,
        "verifier_only": False,
        "hidden_reference_required": False,
        "failure_modes": [],
        "partial_credit_rules": [],
        "edge_cases": [],
        "anti_cheat_notes": [],
        "metadata_json": {},
    }
    rubric_score = {
        "rubric_item_id": "ri_1",
        "criterion_name": "artifact",
        "score": 1.0,
        "max_score": 1.0,
        "passed": True,
        "evidence_refs": ["artifacts/hello.txt"],
        "failure_mode": None,
        "notes": "present",
        "confidence": 0.9,
        "artifact_presence_ok": True,
        "semantic_correctness_score": 1.0,
        "reasoning_summary": "The artifact is present.",
        "improvement_suggestion": None,
    }
    return {
        'intake_agent': (_json_smoke_prompt("task intake smoke", {
            "task_intake_spec": {
                "task_id": "smoke_task",
                "normalized_title": "Hello artifact smoke",
                "normalized_instruction": "Create hello.txt containing hello.",
                "domain": "software",
                "subdomain": "certification",
                "professional_role": "QA engineer",
                "apprenticeship_role": "QA engineer",
                "task_family": "file_creation",
                "expected_economic_value": "$50-$100",
                "expected_economic_value_for_agent_apprentice": "$5-$15",
                "workflow_type": "file_creation",
                "skill_targets": ["artifact_creation"],
                "difficulty_tier": "easy",
                "expected_human_deliverable": "hello.txt",
                "expected_agent_deliverable": "artifacts/hello.txt",
                "input_requirements": [],
                "output_requirements": ["hello.txt contains hello"],
                "required_context": [],
                "assumptions": [],
                "constraints": [],
                "allowed_tools": ["filesystem"],
                "disallowed_tools": [],
                "privacy_classification": "synthetic",
                "license": None,
                "allowed_use": "open_research",
                "rubricability_score": 0.9,
                "verifiability_score": 0.9,
                "artifactability_score": 1.0,
                "needs_expert_review": False,
                "metadata_json": {},
            },
            "task_intake_quality_report": {
                "task_id": "smoke_task",
                "instruction_clarity_score": 0.9,
                "input_completeness_score": 0.9,
                "output_contract_score": 1.0,
                "rubricability_score": 0.9,
                "verifiability_score": 0.9,
                "artifactability_score": 1.0,
                "privacy_risk_score": 0.0,
                "license_risk_score": 0.0,
                "ambiguity_score": 0.1,
                "overall_intake_quality_score": 0.9,
                "quality_flags": [],
                "blockers": [],
                "recommended_fix": None,
                "metadata_json": {},
            },
        }), _IntakeSmokeOutput),
        'rubric_agent': (_json_smoke_prompt("rubric smoke", {
            "rubric_spec": {
                "rubric_id": "smoke_rubric",
                "task_id": "smoke_task",
                "task_family_id": None,
                "rubric_version": "v1",
                "rubric_items": [rubric_item],
                "total_weight": 1.0,
                "pass_threshold": 0.7,
                "worker_visible_rubric_ref": "rubric/worker_visible_rubric.md",
                "verifier_private_rubric_ref": "rubric/rubric.json",
                "hidden_reference_policy": "none",
                "scoring_aggregation": "weighted_sum",
                "required_artifacts": ["hello.txt"],
                "disqualifying_errors": [],
                "partial_credit_allowed": True,
                "grader_kind": "hybrid",
                "rubric_generation_source": "agent_assisted",
                "rubric_generation_agent_provider": None,
                "rubric_generation_agent_model": None,
                "rubric_generation_confidence": 0.9,
                "metadata_json": {},
            },
            "rubric_quality_report": {
                "rubric_id": "smoke_rubric",
                "task_id": "smoke_task",
                "criteria_count": 1,
                "total_weight": 1.0,
                "weights_sum_valid": True,
                "has_observable_evidence": True,
                "has_required_artifacts": True,
                "has_partial_credit_rules": False,
                "has_disqualifying_errors": False,
                "has_hidden_reference_policy": True,
                "has_worker_visible_view": True,
                "has_verifier_private_view": True,
                "ambiguous_criteria_count": 0,
                "unverifiable_criteria_count": 0,
                "rubric_quality_score": 0.9,
                "quality_flags": [],
                "blockers": [],
                "metadata_json": {},
            },
        }), _RubricSmokeOutput),
        'grader_agent': (_json_smoke_prompt("grader smoke", {
            "grader_result_id": "smoke_grader",
            "task_id": "smoke_task",
            "attempt_id": "smoke_attempt",
            "attempt_kind": "baseline",
            "rubric_id": "smoke_rubric",
            "grader_kind": "model",
            "score_source": "model_judged",
            "score": 1.0,
            "max_score": 1.0,
            "passed": True,
            "rubric_item_scores": [rubric_score],
            "failed_criteria": [],
            "passed_criteria": ["artifact"],
            "evidence_refs": ["artifacts/hello.txt"],
            "confidence": 0.9,
            "reasoning_summary": "The expected artifact is present.",
            "limitations": [],
            "hidden_reference_used": False,
            "hidden_reference_leaked": False,
            "artifact_contract_score": 1.0,
            "semantic_score": 1.0,
            "model_score": 1.0,
            "legacy_semantic_score": None,
            "legacy_score_source": None,
            "final_score": 1.0,
            "model": None,
            "provider": None,
            "deterministic_precheck_ref": None,
            "llm_prompt_ref_internal": None,
            "llm_response_ref_internal": None,
            "public_prompt_hash": None,
            "public_response_summary": "passed",
            "score_reliability": "verified",
            "verifier_status": "verified",
            "verifier_confidence": 0.9,
            "verifier_issue_count": 0,
            "verifier_issues_summary": None,
            "metadata_json": {},
        }), GraderResult),
        'verifier_agent': (_json_smoke_prompt("verifier smoke", {
            "verifier_result_id": "smoke_verifier",
            "task_id": "smoke_task",
            "attempt_id": "smoke_attempt",
            "attempt_kind": "baseline",
            "grader_result_id": "smoke_grader",
            "verification_status": "verified",
            "artifact_contract_ok": True,
            "evidence_grounding_ok": True,
            "score_consistency_ok": True,
            "hidden_reference_leaked": False,
            "issues": [],
            "confidence": 0.9,
            "verifier_notes": "verified",
            "semantic_evidence_grounding_ok": True,
            "unsupported_claims": [],
            "leakage_check_ok": True,
            "model": None,
            "provider": None,
            "metadata_json": {},
        }), VerifierResult),
        'evaluator_agent': (_json_smoke_prompt("evaluator smoke", {
            "evaluator_feedback": {
                "feedback_id": "smoke_feedback",
                "task_id": "smoke_task",
                "attempt_id": "smoke_attempt",
                "target_actor": "worker",
                "feedback_type": "other",
                "failed_rubric_items": [],
                "evidence_refs": ["artifacts/hello.txt"],
                "artifact_refs": ["artifacts/hello.txt"],
                "feedback_summary": "The attempt passes.",
                "actionable_feedback": [],
                "suggested_revision": "No revision needed.",
                "revision_priority": "low",
                "confidence": 0.9,
                "hidden_reference_used": False,
                "hidden_reference_leaked": False,
                "failed_or_weak_rubric_items": [],
                "artifact_specific_comments": [],
                "trace_specific_comments": [],
                "revision_plan": None,
                "model": None,
                "provider": None,
                "metadata_json": {},
            },
            "revision_plan": {
                "revision_plan_id": "smoke_revision",
                "task_id": "smoke_task",
                "source_attempt_id": "smoke_attempt",
                "target_attempt_id": "smoke_attempt_revised",
                "revision_kind": "local_fix",
                "revision_reason": "No revision needed for smoke.",
                "failed_rubric_items": [],
                "planned_changes": [],
                "expected_score_improvement": 0.0,
                "risk_of_regression": "low",
                "uses_evaluator_feedback": True,
                "metadata_json": {},
            },
        }), _EvaluatorSmokeOutput),
    }

def run_llm_smoke(out_dir: Path, provider_id: str | None = None) -> dict[str, Any]:
    provider_id=_configured_provider(provider_id)
    status=get_model_provider_status(provider_id)
    counters={
        'mentor_model_provider': provider_id,
        'mentor_model_provider_api_key_visible': bool(status['api_key_visible']),
        'mentor_model_provider_client_constructed_ok': bool(status['client_constructed_ok']),
        'mentor_model_provider_available': bool(status['provider_available']),
    }
    if provider_id == "openai":
        openai_status=get_openai_status()
        counters.update({
            'openai_sdk_import_ok': bool(openai_status['openai_sdk_import_ok']),
            'openai_api_key_visible': bool(openai_status['openai_api_key_visible']),
            'openai_client_constructed_ok': bool(openai_status['openai_client_constructed_ok']),
            'openai_available': bool(openai_status['openai_available']),
        })
    secret_ok=True
    for role,(prompt,model) in smoke_prompts().items():
        short=role.replace('_agent','')
        try:
            rr=run_structured_role(role, prompt, model, out_dir/role, allow_fallback=True, provider_override=provider_id)
        except Exception as exc:
            rr=RoleResult(role=role, provider=provider_id, model=_provider_model(provider_id), live_call_ok=False, structured_output_validation_ok=False, prompt_ref='', output_ref='', parsed_output_ref='', error_type=type(exc).__name__, error_message=_safe_error_message(exc), metadata_json={})
        counters[f'{short}_live_call_ok']=bool(rr.live_call_ok)
        counters[f'{short}_structured_output_validation_ok']=bool(rr.structured_output_validation_ok)
        if rr.error_type:
            counters[f'{short}_error_type']=rr.error_type
            counters[f'{short}_error_message']=rr.error_message
        # ensure generated role files do not contain secrets
        if (out_dir/role).exists():
            for p in (out_dir/role).rglob('*'):
                if p.is_file() and contains_secret(p.read_text(errors='ignore')):
                    secret_ok=False
    counters['secret_scan_ok']=secret_ok
    return counters

def format_smoke_counters(counters: dict[str, Any]) -> str:
    lines=[]
    for k,v in counters.items():
        if isinstance(v,bool): v=_bool(v)
        lines.append(f'{k}={v}')
    return '\n'.join(lines)
