from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import os
import tomllib

SUPPORTED={'.pdf','.docx','.doc','.html','.htm','.txt','.md'}


@dataclass
class FileEntry:
    path: Path
    relative_path: str
    roles: list[str]
    digest: str
    size: int
    error: str | None = None


def file_hash(path):
    digest=sha256()
    with path.open('rb') as file:
        for part in iter(lambda:file.read(1024*1024),b''):digest.update(part)
    return digest.hexdigest()


def load_config(path):
    path=Path(path).resolve()
    with path.open('rb') as f:cfg=tomllib.load(f)
    cfg['root']=(path.parent/cfg['root']).resolve(strict=True)
    cfg['tokenizer_cache']=str((path.parent/cfg.get('tokenizer_cache','.cache/tokenizer')).resolve())
    if not cfg['root'].is_dir():raise ValueError('Data root must be an existing directory')
    if not cfg.get('folders'):raise ValueError('Configure role folders in ingest_files.toml')
    for folder,roles in cfg['folders'].items():
        if '/' in folder or '\\' in folder or folder in ('.','..') or not roles or not all(isinstance(r,str) and r for r in roles):
            raise ValueError('Each top-level folder must have a nonempty list of role names')
    size=cfg.get('chunk_tokens',500);overlap=cfg.get('overlap_tokens',80)
    if not 0<=overlap<size:raise ValueError('Require 0 <= overlap_tokens < chunk_tokens')
    if not 1<=cfg.get('embedding_batch_size',16)<=128:raise ValueError('Batch size must be 1..128')
    if not 1<=cfg.get('file_workers',2)<=16:raise ValueError('File workers must be 1..16')
    return cfg


def scan(cfg):
    root=cfg['root'];entries=[];errors=[]
    def onerror(error):errors.append(str(error))
    for directory,dirs,files in os.walk(root,followlinks=False,onerror=onerror):
        dirs[:]=[name for name in dirs if not (Path(directory)/name).is_symlink()
                 and not getattr(Path(directory)/name,'is_junction',lambda:False)()]
        for name in sorted(files):
            path=Path(directory)/name;rel=path.relative_to(root).as_posix()
            if name.startswith(('~$','.')):continue
            roles=cfg['folders'].get(Path(rel).parts[0],[]) if len(Path(rel).parts)>1 else []
            error=None;digest='';size=0
            try:
                if path.is_symlink() or not path.resolve().is_relative_to(root):raise ValueError('Symlink or outside-root path')
                size=path.stat().st_size
                if size>cfg.get('max_file_mb',200)*1024*1024:raise ValueError('File exceeds max_file_mb')
                if not roles:raise ValueError('Unmapped top-level folder')
                if path.suffix.lower() not in SUPPORTED:raise ValueError('Unsupported file extension')
                digest=file_hash(path)
            except (OSError,ValueError) as exc:error=str(exc)
            entries.append(FileEntry(path,rel,roles,digest,size,error))
    return sorted(entries,key=lambda e:e.relative_path),errors


def classify(entry,previous,duplicate=False,signature=None,force=False):
    if entry.error:return 'skipped'
    if previous is None:return 'duplicate' if duplicate else 'new'
    if not force and previous.get('indexed_hash')==entry.digest and previous.get('pipeline_signature')==signature:
        if sorted(previous['role_names'])!=sorted(entry.roles) or previous['status']=='missing':return 'permissions_changed'
        return 'unchanged'
    if duplicate and not force:return 'duplicate'
    return 'changed' if previous.get('indexed_hash') else 'pending'
