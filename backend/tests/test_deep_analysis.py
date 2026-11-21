import json
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.analysis import DeepAnalysis
from app.chat_api import ChatService,MessageRequest
from app.config import Settings
from app.generation import Generator
from app.main import create_app
from app.memory import ConversationMemory
from app.schemas import Chunk
from test_chats import ADMIN,Store


def setup(settings=None,assessments=None,verdicts=None,empty=False,bad_plan=False,draft_text='Вместимость — 20 человек [1].'):
    cfg=settings or Settings(_env_file=None,memory_enabled=False,llm_provider='lmstudio')
    store=Store();calls=[];searches=[]
    assessment_queue=list(assessments or [{'sufficient':True}])
    verdict_queue=list(verdicts or [{'supported':True}])
    def handler(request):
        body=json.loads(request.content);calls.append(body)
        system=body['messages'][0]['content']
        if body.get('stream'):
            entries=[{'choices':[{'delta':{'content':draft_text},'finish_reason':None}]},
                     {'choices':[{'delta':{},'finish_reason':'stop'}]}]
            return httpx.Response(200,text=''.join('data: '+json.dumps(item)+'\n\n' for item in entries))
        if system.startswith('Предложи'):text='invalid' if bad_plan else json.dumps({'queries':['Дополнительный поиск']})
        elif system.startswith('Достаточно'):
            text=json.dumps(assessment_queue.pop(0) if len(assessment_queue)>1 else assessment_queue[0])
        elif system.startswith('Проверь'):
            verdict=verdict_queue.pop(0) if len(verdict_queue)>1 else verdict_queue[0]
            if verdict=='truncated':
                return httpx.Response(200,json={'choices':[{'message':{'content':'{"supported":'},'finish_reason':'length'}]})
            text=json.dumps(verdict)
        else:text=json.dumps({'query':'Уточнённый вопрос'})
        return httpx.Response(200,json={'choices':[{'message':{'content':text},'finish_reason':'stop'}]})
    http=httpx.Client(transport=httpx.MockTransport(handler))
    class Embeddings:
        def encode(self,query,**kwargs):return [1.]*1024
    class Retriever:
        def search(self,uid,query,vector,limit,**kwargs):
            searches.append((uid,query,kwargs))
            return [] if empty else [Chunk(chunk_id=1000+uid,document_id=100+uid,title='Зал',url='https://example.com',content='Вместимость зала — 20 человек.',rrf_score=.03)]
    services=SimpleNamespace(embeddings=Embeddings(),retriever=Retriever(),generator=Generator(cfg,http),chat_store=store)
    service=ChatService(services,store,4,ConversationMemory(cfg,services.generator))
    return cfg,store,services,service,calls,searches,http


def run(service,store,uid=1,body=None):
    chat=store.create(uid,'Test')
    body=body or MessageRequest(user_id=uid,message='Какова вместимость?',request_id=uuid4(),deep_thinking=True)
    prepared=service.prepare(chat['id'],body)
    events=list(service.run(body,*prepared))
    return events,prepared[0],body


def test_deep_mode_retrieves_again_and_validates_before_done():
    cfg,store,services,service,calls,searches,http=setup(assessments=[
        {'sufficient':False,'next_query':'Вместимость второго зала'}, {'sufficient':True}])
    try:
        events,turn,body=run(service,store)
        names=[name for name,data in events]
        result=events[-1][1]
        assert len(searches)==2 and all(uid==1 for uid,q,options in searches)
        assert all('deadline' in options for uid,q,options in searches)
        assert result['analysis']['validation']=='passed' and result['deep_thinking'] is True
        assert result['analysis']['searches']==2
        assert names.index('delta')<names.index('done')
        assert any(name=='stage' and data['stage']=='validate' for name,data in events)
        assert store.turns[turn['id']]['analysis']['mode']=='deep'
        assert store.turns[turn['id']]['status']=='completed'
    finally:http.close()


def test_validation_failure_retries_and_resets_the_provisional_answer():
    cfg,store,services,service,calls,searches,http=setup(verdicts=[
        {'supported':False,'issues':['Число не подтверждено'],'correction_query':'Точная вместимость'},
        {'supported':True}])
    try:
        events,_,_=run(service,store)
        assert any(name=='reset' for name,data in events)
        result=events[-1][1]
        assert result['analysis']['retries']==1 and result['analysis']['validation']=='passed'
        assert len(searches)==2
    finally:http.close()


def test_unconfirmed_answer_is_not_published_after_retry_limit():
    cfg,store,services,service,calls,searches,http=setup(verdicts=[{'supported':False,'issues':['Нет подтверждения']}])
    try:
        events,turn,_=run(service,store)
        result=events[-1][1]
        assert result['analysis']['status']=='validation_failed'
        assert result['sources']==[] and result['analysis']['retries']==1
        assert '20 человек' not in store.turns[turn['id']]['answer']
    finally:http.close()


def test_step_budget_prevents_unbounded_repairs():
    cfg=Settings(_env_file=None,memory_enabled=False,llm_provider='lmstudio',deep_max_steps=4)
    _,store,_,service,_,_,http=setup(settings=cfg,verdicts=[{'supported':False}])
    try:
        events,_,_=run(service,store)
        result=events[-1][1]
        assert result['analysis']['steps']<=4
        assert result['analysis']['status']=='budget_exhausted' and result['sources']==[]
    finally:http.close()


def test_wall_budget_exhausted_before_first_call_does_not_invoke_models():
    cfg,store,services,service,calls,searches,http=setup()
    try:
        chat=store.create(1,'Test');body=MessageRequest(user_id=1,message='тест',request_id=uuid4(),deep_thinking=True)
        turn,_,_=service.prepare(chat['id'],body);turn['_started']-=cfg.deep_timeout_seconds+1
        events=list(service.run(body,turn,None,False))
        assert events[-1][1]['analysis']['status']=='budget_exhausted'
        assert not calls and not searches
    finally:http.close()


def test_empty_retrieval_abstains_without_draft_or_verifier():
    cfg,store,services,service,calls,searches,http=setup(empty=True)
    try:
        events,_,_=run(service,store)
        assert events[-1][1]['analysis']['status']=='no_sources'
        assert not any(body.get('stream') for body in calls)
        assert len(searches)<=cfg.deep_max_searches
    finally:http.close()


def test_invalid_plan_falls_back_and_request_remains_scoped_to_user():
    cfg,store,services,service,calls,searches,http=setup(bad_plan=True)
    try:
        events,_,_=run(service,store,uid=2)
        assert events[-1][1]['sources'][0]['document_id']==102
        assert all(uid==2 for uid,q,options in searches)
        assert events[-1][1]['analysis']['warnings']
    finally:http.close()


def test_cancelled_deep_request_releases_slot_and_never_calls_model():
    cfg,store,services,service,calls,searches,http=setup()
    try:
        chat=store.create(1,'Test');body=MessageRequest(user_id=1,message='тест',request_id=uuid4(),deep_thinking=True)
        prepared=service.prepare(chat['id'],body)
        store.turns[prepared[0]['id']]['cancel_requested']=True
        events=list(service.run(body,*prepared))
        assert events[-1][0]=='cancelled' and not calls
        assert store.turns[prepared[0]['id']]['status']=='cancelled'
    finally:http.close()


def test_api_persists_mode_and_rejects_mode_change_with_same_request_id():
    cfg=Settings(_env_file=None,memory_enabled=False,llm_provider='lmstudio',dev_auth_enabled=True,
                 dev_api_token='regular-test-token-with-32-characters',dev_admin_token=ADMIN)
    cfg,store,services,service,calls,searches,http=setup(settings=cfg)
    try:
        with TestClient(create_app(cfg,services)) as client:
            client.headers['Authorization']='Bearer '+ADMIN
            cid=client.post('/dev/conversations',json={'user_id':1}).json()['id']
            body={'user_id':1,'message':'Вместимость?','request_id':str(uuid4()),'deep_thinking':True}
            response=client.post(f'/dev/conversations/{cid}/messages',json=body)
            assert response.status_code==200 and response.json()['analysis']['validation']=='passed'
            replay=client.post(f'/dev/conversations/{cid}/messages',json=body)
            assert replay.json()['replayed'] is True
            assert client.post(f'/dev/conversations/{cid}/messages',json={**body,'deep_thinking':False}).status_code==409
            assert client.post(f'/dev/conversations/{cid}/messages',json={**body,'request_id':str(uuid4()),'user_id':2}).status_code==404
            assert client.post(f'/dev/conversations/{cid}/messages',json={**body,'max_steps':999}).status_code==422
            history=client.get(f'/dev/conversations/{cid}/messages?user_id=1').json()
            assert history[0]['deep_thinking'] is True and history[0]['analysis']['mode']=='deep'
            streamed=client.post(f'/dev/conversations/{cid}/messages/stream',json={**body,'request_id':str(uuid4())})
            assert streamed.status_code==200 and 'event: stage' in streamed.text
            assert 'event: search' in streamed.text and 'event: done' in streamed.text
    finally:http.close()


def test_truncated_verifier_is_a_failed_check_not_http_failure():
    cfg,store,services,service,calls,searches,http=setup(verdicts=['truncated'])
    try:
        events,turn,_=run(service,store)
        result=events[-1][1]
        assert result['analysis']['status']=='validation_failed' and result['sources']==[]
        assert result['analysis']['warnings'] and store.turns[turn['id']]['status']=='completed'
    finally:http.close()


def test_search_limit_is_enforced_with_an_insufficient_assessment():
    cfg,store,services,service,calls,searches,http=setup(assessments=[
        {'sufficient':False,'next_query':'Second'}, {'sufficient':False,'next_query':'Third'},
        {'sufficient':False,'next_query':'Fourth'}])
    try:
        events,_,_=run(service,store)
        assert len(searches)==3 and events[-1][1]['analysis']['searches']==3
    finally:http.close()


def test_deep_model_and_schema_are_request_local():
    cfg=Settings(_env_file=None,memory_enabled=False,llm_provider='lmstudio',deep_llm_model='strong-model',
                 deep_llm_base_url='http://strong-model/v1',llm_api_key='private-normal-key')
    _,store,services,service,calls,_,http=setup(settings=cfg)
    try:
        run(service,store)
        assert all(payload['model']=='strong-model' for payload in calls)
        assert any('response_format' in payload for payload in calls)
        assert services.generator.settings.llm_model==cfg.llm_model
        scoped=services.generator.scoped(cfg.model_copy(update={'llm_api_key':cfg.deep_llm_api_key}),999999999)
        assert scoped.settings.llm_api_key.get_secret_value()==''
    finally:http.close()


def test_expired_generator_deadline_does_not_start_http_call():
    from app.errors import ServiceError
    cfg=Settings(_env_file=None)
    def handler(request):raise AssertionError('No HTTP call after deadline')
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        generator=Generator(cfg,http,deadline=0)
        with pytest.raises(ServiceError,match='budget exhausted'):
            generator.complete_text([{'role':'user','content':'test'}])



def test_truncated_deep_draft_retries_with_a_shorter_answer_instruction():
    from app.errors import IncompleteModelOutput
    cfg,store,services,service,calls,searches,http=setup()
    original=services.generator.stream_answer
    def stream(*args,**kwargs):
        if not getattr(stream,'retried',False):
            stream.retried=True
            yield 'delta',{'text':'Оборванный черновик'}
            raise IncompleteModelOutput('token limit',502)
        assert 'до 5 предложений' in kwargs['answer_instruction']
        yield from original(*args,**kwargs)
    services.generator.stream_answer=stream
    # DeepAnalysis clones Generator for request-local settings, so override the clone as well.
    original_scoped=services.generator.scoped
    def scoped(*args,**kwargs):
        clone=original_scoped(*args,**kwargs);clone.stream_answer=stream;return clone
    services.generator.scoped=scoped
    try:
        events,_,_=run(service,store)
        assert events[-1][1]['analysis']['validation']=='passed'
        assert events[-1][1]['analysis']['retries']==1
        assert any(name=='reset' for name,data in events)
    finally:http.close()



def test_related_information_and_qualified_hypothesis_are_returned_as_partial_answer():
    text='В описанном зале 20 мест [1]. Возможно, стоит ориентироваться на группу до 20 гостей, если речь об этом зале; прямой рекомендации в документе нет.'
    cfg,store,services,service,calls,searches,http=setup(
        verdicts=[{'supported':True,'partial':True}],draft_text=text)
    try:
        body=MessageRequest(user_id=1,message='Сколько гостей пригласить?',request_id=uuid4(),deep_thinking=True)
        events,turn,_=run(service,store,body=body)
        result=events[-1][1]
        assert result['answer']==text and result['sources']
        assert result['analysis']['validation']=='partial' and result['analysis']['retries']==0
        assert store.turns[turn['id']]['answer']==text
        assert len(searches)==1 and not any(name=='reset' for name,data in events)
    finally:http.close()


def test_partial_label_does_not_bypass_rejection_of_invented_facts():
    cfg,store,services,service,calls,searches,http=setup(
        verdicts=[{'supported':False,'partial':True,'issues':['Число 500 противоречит вместимости']}],
        draft_text='Возможно, в зале 500 мест [1].')
    try:
        events,turn,_=run(service,store)
        assert events[-1][1]['analysis']['status']=='validation_failed'
        assert '500' not in events[-1][1]['answer'] and not events[-1][1]['sources']
    finally:http.close()
