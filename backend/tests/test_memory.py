import json
from unittest.mock import Mock
from uuid import uuid4

import httpx
import pytest

from app.config import Settings
from app.errors import ServiceError
from app.generation import Generator
from app.memory import ConversationMemory, bounded_history
from app.schemas import Chunk
from app.chat_api import ChatService, MessageRequest
from test_chats import Store, make_services


def row(**changes):
    return {'id':uuid4(),'status':'completed','question':'Какие залы доступны?',
            'answer':'Зал А и зал Б. [7]','evidence_document_ids':[15],'redacted':False,**changes}


def test_bounded_memory_keeps_complete_pairs_and_removes_old_citations():
    rows=[row(question=f'Вопрос {i}') for i in range(9)]
    messages,ids,docs=bounded_history(rows,max_turns=2,max_chars=6000)
    assert [m['content'] for m in messages if m['role']=='user']==['Вопрос 7','Вопрос 8']
    assert '[7]' not in str(messages)
    assert ids==[rows[7]['id'],rows[8]['id']] and docs==[15]


def test_failed_cancelled_and_revoked_answers_are_excluded():
    messages,ids,docs=bounded_history([row(status='failed'),row(status='cancelled'),
                                     row(redacted=True),row(answer='')])
    assert messages==ids==docs==[]


def test_memory_budget_never_splits_a_pair():
    messages,ids,docs=bounded_history([row(),row(answer='x'*1500)],max_chars=100)
    assert messages==[]


def test_followup_is_rewritten_using_only_given_history():
    generator=Mock()
    generator.complete_text.return_value='{"query":"Какова вместимость зала Б?"}'
    memory=ConversationMemory(Settings(_env_file=None,memory_summary_enabled=False),generator)
    context=memory.resolve('А сколько человек во втором?',[row()])
    assert context.query=='Какова вместимость зала Б?'
    assert context.status=='rewritten' and context.document_ids==[15]
    payload=json.loads(generator.complete_text.call_args.args[0][1]['content'])
    assert payload['question']=='А сколько человек во втором?'
    assert len(payload['history'])==2


@pytest.mark.parametrize('enabled,rows,status',[(False,[row()],'disabled'),(True,[],'no_history')])
def test_disabled_or_empty_memory_skips_llm(enabled,rows,status):
    generator=Mock()
    context=ConversationMemory(Settings(_env_file=None,memory_enabled=enabled),generator).resolve('Вопрос',rows)
    assert context.query=='Вопрос' and context.status==status
    generator.complete_text.assert_not_called()


@pytest.mark.parametrize('response',['Not JSON','{"query":""}','{"query":42}','[]'])
def test_invalid_rewrite_falls_back_to_original(response):
    generator=Mock();generator.complete_text.return_value=response
    context=ConversationMemory(Settings(_env_file=None,memory_summary_enabled=False),generator).resolve('Исходный вопрос',[row()])
    assert context.status=='fallback' and context.query=='Исходный вопрос'
    assert context.warning and context.messages


def test_rewrite_network_error_falls_back_without_leaking_upstream_details():
    generator=Mock();generator.complete_text.side_effect=ServiceError('upstream secret')
    result=ConversationMemory(Settings(_env_file=None,memory_summary_enabled=False),generator).resolve('Вопрос',[row()])
    assert result.status=='fallback' and 'secret' not in result.warning


def test_lmstudio_rewrite_contract_and_answer_context():
    cfg=Settings(_env_file=None,llm_provider='lmstudio',llm_base_url='http://model/v1')
    def handler(request):
        payload=json.loads(request.content)
        assert payload['reasoning_effort']=='none' and payload['stream'] is False
        assert payload['max_tokens']==384
        return httpx.Response(200,json={'choices':[{'message':{'content':'{"query":"Зал Б: вместимость"}'},'finish_reason':'stop'}]})
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        generator=Generator(cfg,http)
        context=ConversationMemory(cfg,generator).resolve('А во втором?',[row()])
        sample=Chunk(chunk_id=1,document_id=2,title='Вместимость',url='https://example.com',content='Зал Б: 20 мест',rrf_score=.03)
        path,payload,sources=generator.request_data('А во втором?',[sample],history=context.messages,search_query=context.query)
        assert 'Исходный вопрос пользователя: А во втором?' in payload['messages'][1]['content']
        assert 'Зал Б: вместимость' in payload['messages'][1]['content']
        assert 'История для понимания' in payload['messages'][1]['content']
        assert '[7]' not in payload['messages'][1]['content']
        assert sources[0].document_id==2


def test_pipeline_embeds_rewritten_query_and_keeps_original_question_and_provenance():
    class RecordedStore(Store):
        def evidence(self,uid,tid,docs):self.turns[tid]['evidence_document_ids']=docs
    store=RecordedStore();services=make_services(store)
    services.embeddings.encode=Mock(return_value=[1.]*1024)
    services.retriever.search=Mock(return_value=[Chunk(chunk_id=1,document_id=2,title='Test',url='https://example.com',content='Test',rrf_score=.03)])
    services.generator.complete_text=Mock(return_value='{"query":"Вместимость зала Б"}')
    service=ChatService(services,store,4,ConversationMemory(Settings(_env_file=None,memory_summary_enabled=False),services.generator))
    chat=store.create(1,'A');other=store.create(2,'B')
    first=MessageRequest(user_id=1,message='Какие залы?',request_id=uuid4())
    initial,_=store.begin(1,chat['id'],first)
    store.evidence(1,initial['id'],[15]);store.finish(1,initial['id'],'completed',answer='Зал А, зал Б [1]')
    question=MessageRequest(user_id=1,message='А во втором?',request_id=uuid4())
    prepared=service.prepare(chat['id'],question)
    assert services.embeddings.encode.call_args.args==('Вместимость зала Б',)
    assert services.retriever.search.call_args.args[1]=='Вместимость зала Б'
    assert set(store.turns[prepared[0]['id']]['evidence_document_ids'])=={2,15}
    events=list(service.run(question,*prepared))
    done=events[-1][1]
    assert done['memory_turns']==1 and done['retrieval_query']=='Вместимость зала Б'
    assert store.turns[prepared[0]['id']]['question']=='А во втором?'
    # A different user's chat starts without memory and does not call the rewrite LLM.
    services.generator.complete_text.reset_mock()
    other_body=MessageRequest(user_id=2,message='А во втором?',request_id=uuid4())
    other_prepared=service.prepare(other['id'],other_body)
    services.generator.complete_text.assert_not_called()
    assert other_prepared[0]['memory_turns']==0
    list(service.run(other_body,*other_prepared))


@pytest.mark.parametrize('separate_url',[False,True])
def test_summary_is_rolling_scoped_and_used_for_rewrite_and_answer(separate_url):
    cfg=Settings(_env_file=None,memory_summary_enabled=True,llm_provider='lmstudio',
                 llm_model='main-qwen',llm_base_url='http://main/v1',llm_api_key='main-secret',
                 summary_llm_model='summary-qwen',summary_llm_base_url='http://summary/v1' if separate_url else '')
    calls=[]
    def handler(request):
        payload=json.loads(request.content);calls.append(payload)
        if payload['model']=='summary-qwen':
            assert str(request.url).startswith('http://summary' if separate_url else 'http://main')
            assert request.headers.get('Authorization')==(None if separate_url else 'Bearer main-secret')
            data=json.loads(payload['messages'][1]['content'])
            assert data['previous_summary']=='Пользователь выбирает зал для встречи.'
            assert len(data['messages'])==2 and 'obsolete' not in str(data)
            text='{"summary":"Пользователь выбрал зал Б на 20 человек."}'
        else:
            assert 'Пользователь выбрал зал Б на 20 человек.' in payload['messages'][1]['content']
            text='{"query":"Вместимость зала Б"}'
        assert payload['reasoning_effort']=='none'
        return httpx.Response(200,json={'choices':[{'message':{'content':text},'finish_reason':'stop'}]})
    prior=row(memory_summary='Пользователь выбирает зал для встречи.',summary_status='completed',
              question='Выбираю зал Б',answer='Зал Б на 20 человек [1]',evidence_document_ids=[15,16])
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        generator=Generator(cfg,http)
        result=ConversationMemory(cfg,generator).resolve('А вместимость?',[row(question='obsolete'),prior])
        assert result.summary_status=='completed' and result.summary_model=='summary-qwen'
        assert result.document_ids==[15,16] and result.llm_calls==2
        assert result.messages[-1]['content']=='Зал Б на 20 человек'
        assert generator.settings.llm_model=='main-qwen'
        sample=Chunk(chunk_id=1,document_id=2,title='Зал',url='',content='Зал Б: 20 мест',rrf_score=.03)
        _,payload,_=generator.request_data('А вместимость?',[sample],history=result.messages,history_summary=result.summary)
        assert 'Пользователь выбрал зал Б на 20 человек.' in payload['messages'][1]['content']
        assert len(calls)==2


def test_invalid_summary_falls_back_without_reusing_revoked_content():
    cfg=Settings(_env_file=None,memory_summary_enabled=True,llm_provider='lmstudio')
    def handler(request):
        payload=json.loads(request.content)
        assert 'private revoked' not in str(payload)
        text='{"summary":""}' if 'response_format' in payload else '{"query":"Вопрос"}'
        return httpx.Response(200,json={'choices':[{'message':{'content':text},'finish_reason':'stop'}]})
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        result=ConversationMemory(cfg,Generator(cfg,http)).resolve('Вопрос',[
            row(redacted=True,memory_summary='private revoked',summary_status='completed'),row()])
        assert result.summary=='' and result.summary_status=='fallback' and result.warning
        assert result.messages and result.document_ids==[15]


def test_summary_only_history_does_not_leak_between_conversations():
    cfg=Settings(_env_file=None,memory_summary_enabled=True,llm_provider='lmstudio')
    with httpx.Client(transport=httpx.MockTransport(lambda request: (_ for _ in ()).throw(AssertionError('No model call')))) as http:
        context=ConversationMemory(cfg,Generator(cfg,http)).resolve('Вопрос',[])
        assert context.summary=='' and context.summary_status=='no_history' and context.llm_calls==0



def test_chat_pipeline_saves_summary_and_passes_it_to_the_answer_generator():
    cfg=Settings(_env_file=None,memory_summary_enabled=True,llm_provider='lmstudio')
    calls=[]
    def handler(request):
        payload=json.loads(request.content);calls.append(payload)
        if payload.get('stream'):
            assert 'Краткая сводка про зал Б' in payload['messages'][1]['content']
            entry={'choices':[{'delta':{'content':'В зале Б 20 мест [1].'},'finish_reason':'stop'}]}
            return httpx.Response(200,text='data: '+json.dumps(entry)+'\n\n')
        text='{"summary":"Краткая сводка про зал Б"}' if 'response_format' in payload else '{"query":"Вместимость зала Б"}'
        return httpx.Response(200,json={'choices':[{'message':{'content':text},'finish_reason':'stop'}]})
    store=Store();services=make_services(store)
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        services.generator=Generator(cfg,http)
        service=ChatService(services,store,4,ConversationMemory(cfg,services.generator))
        chat=store.create(1,'Summary test')
        initial,_=store.begin(1,chat['id'],MessageRequest(user_id=1,message='Выбираю зал Б',request_id=uuid4()))
        store.finish(1,initial['id'],'completed',answer='В зале Б 20 мест [1]')
        body=MessageRequest(user_id=1,message='А вместимость?',request_id=uuid4())
        prepared=service.prepare(chat['id'],body)
        events=list(service.run(body,*prepared))
        done=events[-1][1]
        assert done['summary_status']=='completed' and done['memory_summary']=='Краткая сводка про зал Б'
        saved=store.turns[prepared[0]['id']]
        assert saved['memory_summary']==done['memory_summary']
        assert saved['status']=='completed' and len(calls)==3



def test_summary_carries_old_evidence_even_when_its_pair_is_outside_input_budget():
    cfg=Settings(_env_file=None,memory_summary_enabled=True,llm_provider='lmstudio',memory_max_chars=1000)
    ancestor=row(question='Old pair',memory_summary='Old summary',summary_status='completed',evidence_document_ids=[99])
    latest=[row(question='q'*500,answer='a'*1000,evidence_document_ids=[15]) for _ in range(8)]
    def handler(request):
        payload=json.loads(request.content)
        if 'response_format' in payload:
            data=json.loads(payload['messages'][1]['content'])
            assert data['previous_summary']=='Old summary'
            assert 'Old pair' not in str(data['messages'])
            text='{"summary":"Updated summary"}'
        else:text='{"query":"Question"}'
        return httpx.Response(200,json={'choices':[{'message':{'content':text},'finish_reason':'stop'}]})
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        result=ConversationMemory(cfg,Generator(cfg,http)).resolve('Question',[ancestor,*latest])
        assert result.summary=='Updated summary' and result.messages==[]
        assert result.document_ids==[15,99] and ancestor['id'] in result.turn_ids
