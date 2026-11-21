"""Opt-in tests against the local Docker PostgreSQL. Only disposable test rows are removed."""
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.chat_api import MessageRequest
from app.chat_store import ChatStore
from app.config import Settings
from app.db import Retriever
from app.memory import MemoryContext

pytestmark=pytest.mark.skipif(os.environ.get('CHAT_DB_TESTS')!='1',reason='Set CHAT_DB_TESTS=1 for local database tests')


def admin(sql):
    result=subprocess.run(['docker','exec','-i',os.environ.get('CHAT_TEST_CONTAINER','testagenticrag-db-1'),
                           'psql','-qAt','-v','ON_ERROR_STOP=1','-U','rag_admin','-d','rag'],
                          input=sql,text=True,encoding='utf-8',capture_output=True,check=True)
    return result.stdout.strip()


@pytest.fixture
def database():
    marker='chat_test_'+uuid4().hex
    uid=int(admin(f"INSERT INTO app_users(auth_subject) VALUES('{marker}_a') RETURNING id;"))
    other=int(admin(f"INSERT INTO app_users(auth_subject) VALUES('{marker}_b') RETURNING id;"))
    doc=int(admin(f"INSERT INTO documents(url,title,content_hash) VALUES('https://example.com/{marker}','Chat test','test') RETURNING id;"))
    role=int(admin(f"INSERT INTO app_roles(name) VALUES('{marker}') RETURNING id;"))
    admin(f"INSERT INTO user_roles VALUES({uid},{role}); INSERT INTO document_roles VALUES({doc},{role});")
    retriever=Retriever(Settings());retriever.open();store=ChatStore(retriever)
    try:yield store,uid,other,doc,role
    finally:
        retriever.close()
        # Exact fixture IDs only; no application-owned conversations are touched.
        admin(f"""BEGIN;
        DELETE FROM rag_turns WHERE conversation_id IN (SELECT id FROM rag_conversations WHERE user_id IN ({uid},{other}));
        DELETE FROM rag_conversations WHERE user_id IN ({uid},{other});
        DELETE FROM document_roles WHERE document_id={doc};
        DELETE FROM user_roles WHERE user_id IN ({uid},{other});
        DELETE FROM documents WHERE id={doc};
        DELETE FROM app_users WHERE id IN ({uid},{other});
        DELETE FROM app_roles WHERE id={role};COMMIT;""")


def test_real_rls_idempotency_and_revoked_sources(database):
    store,uid,other,doc,role=database
    chat=store.create(uid,'Test')
    with pytest.raises(HTTPException) as exc:store.history(other,chat['id'])
    assert exc.value.status_code==404
    body=MessageRequest(user_id=uid,message='test',request_id=uuid4())
    turn,replay=store.begin(uid,chat['id'],body)
    assert not replay
    store.evidence(uid,turn['id'],[doc])
    store.save_memory(uid,turn['id'],MemoryContext(query='Standalone query',turn_ids=[],status='unchanged',summary='Private history summary',summary_status='completed',summary_model='qwen'))
    sources=[{'id':1,'document_id':doc,'chunk_id':1,'title':'Test','url':'https://example.com'}]
    store.finish(uid,turn['id'],'completed',answer='answer [1]',sources=sources)
    assert store.begin(uid,chat['id'],body)[1] is True
    store.save_analysis(uid,turn['id'],{'mode':'deep','queries':['private query'],'validation':'passed'})
    assert store.history(uid,chat['id'])[0]['answer']=='answer [1]'
    assert store.history(uid,chat['id'])[0]['retrieval_query']=='Standalone query'
    assert store.history(uid,chat['id'])[0]['memory_summary']=='Private history summary'
    assert store.memory_history(uid,chat['id'],uuid4(),6)[0]['id']==turn['id']
    with pytest.raises(HTTPException):store.memory_history(other,chat['id'],uuid4(),6)
    admin(f'DELETE FROM document_roles WHERE document_id={doc};')
    history=store.history(uid,chat['id'])
    assert history[0]['redacted'] is True and history[0]['sources']==[]
    assert history[0]['retrieval_query'] is None
    assert history[0]['analysis']=={}
    assert history[0]['memory_summary']=='' and history[0]['summary_status']=='redacted'
    assert store.begin(uid,chat['id'],body)[0]['redacted'] is True
    with store.retriever.pool.connection() as c:
        assert c.execute('SELECT count(*) AS n FROM rag_conversations').fetchone()['n']==0


def test_real_database_busy_chat_and_cancellation(database):
    store,uid,other,doc,role=database
    chat=store.create(uid,'Parallel test')
    def begin():
        try:return store.begin(uid,chat['id'],MessageRequest(user_id=uid,message='test',request_id=uuid4()))[0]
        except HTTPException as exc:return exc.status_code
    with ThreadPoolExecutor(2) as executor:results=list(executor.map(lambda _:begin(),range(2)))
    assert sum(r==409 for r in results if isinstance(r,int))==1
    turn=next(r for r in results if isinstance(r,dict))
    store.cancel(uid,chat['id'],turn['request_id'])
    assert store.cancelled(uid,turn['id'])
    assert store.finish(uid,turn['id'],'completed',answer='partial')=='cancelled'


def test_real_database_keeps_deep_mode_and_rejects_replay_in_another_mode(database):
    store,uid,other,doc,role=database
    chat=store.create(uid,'Deep mode test')
    request_id=uuid4()
    body=MessageRequest(user_id=uid,message='test',request_id=request_id,deep_thinking=True)
    turn,_=store.begin(uid,chat['id'],body)
    assert turn['deep_thinking'] is True
    store.save_analysis(uid,turn['id'],{'mode':'deep','validation':'passed'})
    store.finish(uid,turn['id'],'completed',answer='test')
    assert store.history(uid,chat['id'])[0]['analysis']['validation']=='passed'
    with pytest.raises(HTTPException) as exc:
        store.begin(uid,chat['id'],MessageRequest(user_id=uid,message='test',request_id=request_id,deep_thinking=False))
    assert exc.value.status_code==409
