from __future__ import annotations
import os, re
from pathlib import Path

def parse_env_file(path: Path) -> dict[str,str]:
    vals={}
    if not path.exists(): return vals
    for line in path.read_text().splitlines():
        line=line.strip()
        if not line or line.startswith('#') or '=' not in line: continue
        k,v=line.split('=',1); vals[k.strip()]=v.strip().strip('"').strip("'")
    return vals

def load_local_env(root: Path | None=None) -> dict[str,str]:
    if os.getenv("AA_DISABLE_LOCAL_ENV") == "1":
        return {}
    root = root or Path.cwd()
    loaded={}
    for name in ['.env.local','.env']:
        for k,v in parse_env_file(root/name).items():
            if k not in os.environ:
                os.environ[k]=v; loaded[k]=v
    return loaded

SECRET_PATTERNS=[
    re.compile(r"(?<![A-Za-z0-9])sk-proj-[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?<![A-Za-z0-9])sk-ant-[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?<![A-Za-z0-9])sk-or-v1-[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AIza[A-Za-z0-9_-]{20,}"),
    re.compile(r"dsk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"\buser_(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{16,}\b"),
    re.compile(r"(['\"]user_id['\"]\s*:\s*)['\"][^'\"]+['\"]"),
]
def contains_secret(text: str) -> bool:
    return any(p.search(text or '') for p in SECRET_PATTERNS)
def redact_secrets(text: str) -> str:
    text = text or ''
    for p in SECRET_PATTERNS:
        if 'user_id' in p.pattern:
            text=p.sub(r"\1'[REDACTED_USER_ID]'", text)
        elif p.pattern.startswith('user_'):
            text=p.sub('[REDACTED_USER_ID]', text)
        else:
            text=p.sub('[REDACTED_SECRET]', text)
    return text
