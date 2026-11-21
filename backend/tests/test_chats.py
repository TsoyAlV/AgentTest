from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.chat_api import ChatService, MessageRequest
from app.config import Settings
from app.main import create_app
from app.schemas import Chunk

ADMIN='local-test-admin-token-at-least-32-characters'


class Store:
    def __init__(self):
        self.lock=Lock();self.chats={};self.turns={}
    def users(self):return [{'id':1,'auth_subject':'test-a','roles':['employee']},{'id':2,'auth_subject':'test-b','roles':[]}]
    def create(self,user_id,title):
        row={'id':str(uuid4()),'user_id':user_id,'title':title}
        self.chats[row['id']]=row
        return row
    def conversations(self,user_id):return [c for c in self.chats.values() if c['user_id']==user_id]
    def own(self,user_id,cid):
        row=self.chats.get(str(cid))
        if not row or row['user_id']!=user_id:raise HTTPException(404,'Chat not found')
    def history(self,user_id,cid):
        self.own(user_id,cid)
        return [t for t in self.turns.values() if t['conversation_id']==str(cid)]
    def begin(self,user_id,cid,body):
        with self.lock:
            self.own(user_id,cid)
            rows=self.history(user_id,cid)
            old=next((t for t in rows if t['request_id']==str(body.request_id)),None)
            if old:
                if (old['status']=='completed' and old['question']==body.message and old['client_id']==body.client_id
                        and old.get('deep_thinking',False)==body.deep_thinking):return old,True
                raise HTTPException(409,'Existing request')
            if any(t['status']=='generating' for t in rows):raise HTTPException(409,'Chat busy')
            row={'id':str(uuid4()),'conversation_id':str(cid),'request_id':str(body.request_id),
                 'question':body.message,'client_id':body.client_id,'deep_thinking':body.deep_thinking,
                 'status':'generating','answer':'','sources':[],'cancel_requested':False}
            self.turns[row['id']]=row
            return row,False
    def evidence(self,*args):pass
    def memory_history(self,uid,cid,exclude,max_turns):
        return [t for t in self.history(uid,cid) if t['id']!=exclude and t['status']=='completed'][-max_turns:]
    def save_memory(self,uid,tid,memory):
        self.turns[tid].update(**memory.diagnostics(),memory_turn_ids=memory.turn_ids)
    def save_analysis(self,uid,tid,analysis):self.turns[tid]['analysis']=analysis
    def cancelled(self,uid,tid):return self.turns[tid]['cancel_requested']
    def cancel(self,uid,cid,rid):
        self.own(uid,cid)
        for t in self.history(uid,cid):
            if t['request_id']==str(rid):t['cancel_requested']=True
        return {'cancel_requested':True}
    def finish(self,uid,tid,status,answer='',sources=None,error=None):
        t=self.turns[tid];t.update(status='cancelled' if t['cancel_requested'] else status,answer=answer,sources=sources or [],error=error)
        return t['status']


def make_services(store):
    class Embeddings:
        def encode(self,q):return [1.]*1024
    class Retriever:
        def search(self,uid,*args):return [Chunk(chunk_id=1,document_id=1,title='Test',url='https://example.com',content='test',rrf_score=.03)]
    class Generator:
        def stream_answer(self,*args,**kwargs):
            yield 'delta',{'text':'Ответ [1]'}
            yield 'done',{'answer':'Ответ [1]','sources':[]}
        def complete_text(self,messages,max_tokens=384):return '{"query":"Вопрос"}'
    return SimpleNamespace(embeddings=Embeddings(),retriever=Retriever(),generator=Generator(),chat_store=store)


def app_client():
    store=Store()
    cfg=Settings(_env_file=None,dev_auth_enabled=True,dev_api_token='regular-test-token-at-least-32-characters',dev_admin_token=ADMIN)
    return TestClient(create_app(cfg,make_services(store))),store


def test_chat_api_separates_admin_and_regular_token_and_users():
    client,store=app_client()
    with client:
        assert client.get('/').status_code==200
        assert client.get('/static/chat.js').status_code==200
        assert client.get('/dev/users').status_code==401
        assert client.get('/dev/users',headers={'Authorization':'Bearer regular-test-token-at-least-32-characters'}).status_code==401
        client.headers['Authorization']='Bearer '+ADMIN
        assert len(client.get('/dev/users').json())==2
        c=client.post('/dev/conversations',json={'user_id':1}).json()
        cid=c['id']
        assert client.get('/dev/conversations/'+cid+'/messages?user_id=2').status_code==404
        assert client.get('/dev/users/2/conversations').json()==[]
        body={'user_id':1,'message':'Вопрос','client_id':'test','request_id':str(uuid4())}
        first=client.post('/dev/conversations/'+cid+'/messages',json=body)
        assert first.status_code==200
        assert first.json()['answer']=='Ответ [1]'
        retry=client.post('/dev/conversations/'+cid+'/messages',json=body)
        assert retry.json()['replayed'] is True
        assert len(store.history(1,cid))==1
        changed=client.post('/dev/conversations/'+cid+'/messages',json={**body,'message':'Другой'})
        assert changed.status_code==409
        stream=client.post('/dev/conversations/'+cid+'/messages/stream',json={**body,'request_id':str(uuid4())})
        assert 'event: meta' in stream.text and 'event: delta' in stream.text and 'event: done' in stream.text


def test_parallel_chats_and_busy_chat_and_capacity():
    store=Store();svc=ChatService(make_services(store),store,2)
    a=store.create(1,'A');b=store.create(2,'B');c=store.create(1,'C')
    body=lambda uid:MessageRequest(user_id=uid,message='тест',request_id=uuid4())
    a_body,b_body=body(1),body(2)
    # Two independent chats are admitted at the same time.
    with ThreadPoolExecutor(2) as executor:
        fa=executor.submit(svc.prepare,a['id'],a_body)
        fb=executor.submit(svc.prepare,b['id'],b_body)
        pa,pb=fa.result(),fb.result()
    with pytest.raises(HTTPException) as exc:svc.prepare(c['id'],body(1))
    assert exc.value.status_code==429
    list(svc.run(a_body,*pa))
    with pytest.raises(HTTPException) as exc:svc.prepare(b['id'],body(2))
    assert exc.value.status_code==409
    list(svc.run(b_body,*pb))
    cancel_body=body(1)
    pc=svc.prepare(c['id'],cancel_body)
    # Cancelling before the first model delta preserves the turn as cancelled.
    store.turns[pc[0]['id']]['cancel_requested']=True
    assert any(event=='cancelled' for event,data in svc.run(cancel_body,*pc))
    assert store.turns[pc[0]['id']]['status']=='cancelled'


def test_closed_stream_releases_slot_and_records_cancelled():
    store=Store();svc=ChatService(make_services(store),store,1)
    chat=store.create(1,'A');body=MessageRequest(user_id=1,message='тест',request_id=uuid4())
    prepared=svc.prepare(chat['id'],body)
    iterator=svc.run(body,*prepared)
    assert next(iterator)[0]=='meta'
    iterator.close()
    assert store.turns[prepared[0]['id']]['status']=='cancelled'
    again=MessageRequest(user_id=1,message='тест',request_id=uuid4())
    list(svc.run(again,*svc.prepare(chat['id'],again)))
