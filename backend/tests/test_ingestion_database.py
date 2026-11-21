import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from fastapi import HTTPException

from app.ingestion.files import FileEntry,file_hash
from app.ingestion.chunking import Chunk
from app.ingestion.store import IngestionStore
from app.chat_store import ChatStore
from test_chat_database import database,admin

pytestmark=pytest.mark.skipif(os.environ.get('INGEST_DB_TESTS')!='1',reason='Set INGEST_DB_TESTS=1 and INGEST_DATABASE_URL for import database tests')


def test_atomic_publish_dedup_acl_and_missing_restore(database,tmp_path):
    chat_store,uid,other,doc,role=database
    collection='ingest_test_'+uuid4().hex
    store=IngestionStore(os.environ['INGEST_DATABASE_URL'],collection)
    a=tmp_path/'a.txt';a.write_text('original',encoding='utf-8')
    b=tmp_path/'b.txt';b.write_bytes(a.read_bytes())
    first=FileEntry(a,'employee/a.txt',['employee'],file_hash(a),a.stat().st_size)
    second=FileEntry(b,'other/b.txt',['other'],file_hash(b),b.stat().st_size)
    try:
        source=store.register(first,[role])
        doc1,count=store.publish(source,first,'v1','test-model',[role],
                                 [Chunk('original text','page',{'page_start':1})],[np.ones(1024,dtype=np.float32)])
        assert count==1 and chat_store.document_text(uid,doc1)['chunks'][0]['content']=='original text'
        source2=store.register(second,[])
        doc2,count2=store.publish(source2,second,'v1','test-model',[],copy_document=doc1)
        assert doc2!=doc1 and count2==count
        with pytest.raises(HTTPException):chat_store.document_text(uid,doc2)
        # An invalid replacement must roll back both deletion of old chunks and new rows.
        with pytest.raises(Exception):
            store.publish(source,first,'v1','test-model',[role],[Chunk('broken','',{})],[np.ones(2,dtype=np.float32)])
        assert chat_store.document_text(uid,doc1)['chunks'][0]['content']=='original text'
        store.unavailable([first.relative_path],'missing')
        with pytest.raises(HTTPException):chat_store.document_text(uid,doc1)
        store.touch(first,[role])
        assert chat_store.document_text(uid,doc1)['id']==doc1
        assert store.reusable(first.digest,'v1')['document_id']
        assert store.reusable(first.digest,'v2') is None
    finally:
        admin(f"""BEGIN;
          DELETE FROM chunks WHERE document_id IN (SELECT id FROM documents WHERE metadata->>'collection'='{collection}');
          DELETE FROM document_roles WHERE document_id IN (SELECT id FROM documents WHERE metadata->>'collection'='{collection}');
          DELETE FROM file_sources WHERE collection='{collection}';
          DELETE FROM documents WHERE metadata->>'collection'='{collection}';COMMIT;""")
