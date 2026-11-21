"""Folder reconciliation and per-file atomic indexing. Default mode is read-only plan."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import threading
import getpass

import httpx
import numpy as np

from app.config import Settings
from app.embeddings import Embeddings
from app.ingestion.chunking import chunk_segments,load_tokenizer,signature
from app.ingestion.extract import extract
from app.ingestion.files import scan,load_config,classify,file_hash
from app.ingestion.store import IngestionStore


def execute(args):
    cfg=load_config(args.config);settings=Settings()
    dsn=settings.ingest_database_url.get_secret_value()
    if args.database_user:
        from psycopg.conninfo import conninfo_to_dict,make_conninfo
        params=conninfo_to_dict(settings.database_url.get_secret_value())
        params['user']=args.database_user
        if not args.prompt_password:raise ValueError('Use --prompt-password with --database-user')
        params['password']=getpass.getpass('Database password (not saved): ')
        dsn=make_conninfo(**params)
    store=IngestionStore(dsn,cfg['collection'])
    entries,scan_errors=scan(cfg)
    if scan_errors:raise ValueError('Scan incomplete; nothing was changed: '+'; '.join(scan_errors))
    tokenizer,tokenizer_hash=load_tokenizer(cfg)
    pipeline=signature(cfg,settings,tokenizer_hash)
    with store.run_lock():
        previous=store.snapshot();seen={e.relative_path for e in entries};seen_hashes=set()
        result=[]
        for entry in entries:
            duplicate=entry.digest in seen_hashes or bool(entry.digest and store.reusable(entry.digest,pipeline))
            action=classify(entry,previous.get(entry.relative_path),duplicate,pipeline,args.force)
            result.append({'path':entry.relative_path,'action':action,'sha256':entry.digest,'roles':entry.roles,'error':entry.error})
            if entry.digest:seen_hashes.add(entry.digest)
        missing=sorted(set(previous)-seen)
        report={'collection':cfg['collection'],'created_at':datetime.now(timezone.utc).isoformat(),
                'mode':'apply' if args.apply else 'plan','pipeline_signature':pipeline,
                'files':result,'missing':missing,'counts':dict(Counter(row['action'] for row in result))}
        report_path=Path(args.report);report_path.parent.mkdir(parents=True,exist_ok=True)
        def write_report():report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        if not args.apply:
            write_report();print(json.dumps(report['counts'],ensure_ascii=False));print('Missing files:',len(missing));print('Plan saved:',report_path);return 0
        names={role for e in entries if not e.error for role in e.roles}
        roles=store.roles(names,args.create_roles)
        # Revocations and metadata-only ACL changes do not depend on the model server.
        store.unavailable(missing,'missing')
        store.unavailable([e.relative_path for e in entries if e.error and e.relative_path in previous],'blocked')
        pairs=[(entry,row) for entry,row in zip(entries,result,strict=True) if not entry.error]
        if args.limit:pairs=pairs[:args.limit]
        for entry,row in pairs:
            if row['action'] in ('unchanged','permissions_changed') and not args.force:
                doc,count=store.touch(entry,[roles[name] for name in entry.roles])
                row.update(result='unchanged',document_id=doc,chunks=count)
        write_report()
        with httpx.Client(timeout=settings.http_timeout_seconds,trust_env=False) as client:
            model=Embeddings(settings,client)
            needs_model=any(not row.get('result') and (args.force or not store.reusable(entry.digest,pipeline)) for entry,row in pairs)
            if needs_model:
                try:model.encode('Проверка готовности модели embeddings')
                except Exception as exc:
                    report['apply_error']=type(exc).__name__;write_report();raise
            hash_locks=defaultdict(threading.Lock);print_lock=threading.Lock()
            def process(entry,row):
                if entry.error or row.get('result'):return
                role_ids=[roles[name] for name in entry.roles]
                source=None
                try:
                    with hash_locks[entry.digest]:
                        if row['action'] in ('unchanged','permissions_changed') and not args.force:
                            doc,count=store.touch(entry,role_ids);row.update(result='unchanged',document_id=doc,chunks=count);return
                        source=store.register(entry,role_ids)
                        cached=None if args.force else store.reusable(entry.digest,pipeline)
                        if cached and cached['document_id']!=source['document_id']:
                            if file_hash(entry.path)!=entry.digest:raise ValueError('File changed during indexing; retry')
                            doc,count=store.publish(source,entry,pipeline,settings.embedding_model,role_ids,copy_document=cached['document_id'])
                            row.update(result='reused_embeddings',document_id=doc,chunks=count)
                        else:
                            segments=extract(entry.path,cfg)
                            chunks=chunk_segments(segments,tokenizer,cfg['chunk_tokens'],cfg['overlap_tokens'])
                            vectors=[];batch_size=cfg['embedding_batch_size']
                            for start in range(0,len(chunks),batch_size):
                                batch=chunks[start:start+batch_size]
                                # No filename in model input: renamed byte-identical copies can reuse vectors.
                                vectors.extend(model.encode_many([c.content for c in batch]))
                                if start and start%(batch_size*10)==0:
                                    with print_lock:print(f'Progress {entry.relative_path}: {start}/{len(chunks)} chunks',flush=True)
                            if file_hash(entry.path)!=entry.digest:raise ValueError('File changed during indexing; retry')
                            doc,count=store.publish(source,entry,pipeline,settings.embedding_model,role_ids,chunks,vectors)
                            row.update(result='indexed',document_id=doc,chunks=count)
                    with print_lock:print(f"{row['result']}: {entry.relative_path} ({count} chunks)",flush=True)
                except Exception as exc:
                    # Preserve the previous published version; failure is isolated to this file.
                    error=str(exc) if isinstance(exc,(ValueError,RuntimeError)) else type(exc).__name__
                    from app.errors import ServiceError
                    if isinstance(exc,ServiceError):error=str(exc)
                    row.update(result='failed',error=error)
                    if source:store.failed(source['id'],error)
                    with print_lock:print(f'FAILED {entry.relative_path}: {error}',flush=True)
            with ThreadPoolExecutor(cfg['file_workers']) as executor:
                futures=[executor.submit(process,*pair) for pair in pairs]
                for future in as_completed(futures):
                    future.result();write_report()
            report['results']=dict(Counter(row.get('result','skipped' if row['action']=='skipped' else 'not_processed') for row in result))
            write_report();print(json.dumps(report['results'],ensure_ascii=False));print('Report saved:',report_path)
            return 1 if report['results'].get('failed') else 0


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='ingest_files.toml')
    parser.add_argument('--apply',action='store_true',help='Write changes; without this flag only plan is produced')
    parser.add_argument('--create-roles',action='store_true')
    parser.add_argument('--database-user',help='Use an existing database login for this run')
    parser.add_argument('--prompt-password',action='store_true',help='Prompt for DB password without saving it')
    parser.add_argument('--force',action='store_true',help='Re-extract and re-embed even unchanged files')
    parser.add_argument('--limit',type=int,help='Process at most this many files; scanning is always complete')
    parser.add_argument('--report',default='reports/ingestion.json')
    args=parser.parse_args()
    if args.limit is not None and args.limit<1:parser.error('--limit must be positive')
    try:return execute(args)
    except Exception as exc:
        from app.errors import ServiceError
        detail=str(exc) if isinstance(exc,(ValueError,ServiceError)) else type(exc).__name__
        print('Import stopped:',detail,file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
