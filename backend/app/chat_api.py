import json
import logging
import threading
import time
from uuid import UUID

import anyio
from fastapi import APIRouter, Depends, HTTPException, Request, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator

from .auth import dev_admin
from .errors import ServiceError
from .analysis import DeepAnalysis
from .config import Settings

log=logging.getLogger(__name__)
router=APIRouter(prefix='/dev',tags=['Local user experiments'],dependencies=[Depends(dev_admin)])


class NewConversation(BaseModel):
    user_id: int=Field(gt=0)
    title: str=Field(default='Новый диалог',min_length=1,max_length=120)


class MessageRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    user_id: int=Field(gt=0)
    message: str=Field(min_length=1,max_length=2000)
    client_id: str=Field(default='web-playground',min_length=1,max_length=100)
    request_id: UUID
    limit: int=Field(default=5,ge=1,le=10)
    deep_thinking: StrictBool=False

    @field_validator('message')
    @classmethod
    def trim_message(cls,v):
        if not v.strip():raise ValueError('Message cannot be blank')
        return v.strip()


class CancelRequest(BaseModel):
    user_id: int=Field(gt=0)
    request_id: UUID


class ChatService:
    def __init__(self,services,store,parallelism,memory=None):
        self.services=services
        self.store=store
        self.slots=threading.BoundedSemaphore(parallelism)
        self.memory=memory
        self.settings=memory.settings if memory else Settings(_env_file=None)

    def prepare(self,conversation_id,body):
        started=time.monotonic()
        if not self.slots.acquire(blocking=False):
            raise HTTPException(429,'Generation capacity reached; retry later',headers={'Retry-After':'2'})
        turn=None
        try:
            turn,replay=self.store.begin(body.user_id,conversation_id,body)
            if replay:
                self.slots.release()
                return turn,None,True
            turn={**turn,'_started':started}
            if body.deep_thinking:return turn,None,False
            query=body.message
            memory=None
            if self.memory:
                rows=(self.store.memory_history(body.user_id,conversation_id,turn['id'],20 if self.settings.memory_summary_enabled else self.memory.settings.memory_max_turns)
                      if self.memory.settings.memory_enabled else [])
                memory=self.memory.resolve(body.message,rows)
                query=memory.query
                self.store.save_memory(body.user_id,turn['id'],memory)
                self.store.evidence(body.user_id,turn['id'],memory.document_ids)
                turn={**turn,**memory.diagnostics(),'_memory':memory}
            vector=self.services.embeddings.encode(query)
            chunks=self.services.retriever.search(body.user_id,query,vector,body.limit)
            # Include document provenance from memory to propagate later permission revocations.
            evidence={c.document_id for c in chunks}
            if memory:evidence.update(memory.document_ids)
            self.store.evidence(body.user_id,turn['id'],list(evidence))
            return turn,chunks,False
        except BaseException:
            try:
                if turn:self.store.finish(body.user_id,turn['id'],'failed',error='Request preparation failed')
            finally:self.slots.release()
            raise

    def run(self,body,turn,chunks,replay):
        meta={'conversation_id':str(turn['conversation_id']),'request_id':str(body.request_id),'turn_id':str(turn['id'])}
        diagnostics={'retrieval_query':turn.get('retrieval_query') or body.message,
                     'memory_turns':turn.get('memory_turns',len(turn.get('memory_turn_ids',[]))),
                     'memory_status':turn.get('memory_status','disabled'),
                     'memory_warning':turn.get('memory_warning'),
                     'memory_summary':turn.get('memory_summary',''),
                     'summary_status':turn.get('summary_status','disabled'),
                     'summary_model':turn.get('summary_model')}
        if turn.get('redacted'):
            diagnostics['retrieval_query']=None
            diagnostics['memory_summary']=''
        meta.update(diagnostics)
        meta['deep_thinking']=bool(turn.get('deep_thinking',body.deep_thinking))
        if replay:
            yield 'done',{**meta,'answer':turn['answer'],'sources':turn['sources'],'analysis':turn.get('analysis',{}),'replayed':True}
            return
        parts=[]
        finished=False
        stream=None
        last_cancel_check=0
        try:
            yield 'meta',meta
            yield 'retrieval',diagnostics
            if body.deep_thinking:
                stream=DeepAnalysis(self.settings,self.services,self.store,body,turn).events()
            else:
                memory=turn.get('_memory')
                options={'history':memory.messages,'search_query':memory.query} if memory and memory.messages else {}
                if memory and memory.summary:options['history_summary']=memory.summary
                stream=self.services.generator.stream_answer(body.message,chunks,**options)
            for event,data in stream:
                check_cancel=event=='done' or time.monotonic()-last_cancel_check>=0.4
                if check_cancel:last_cancel_check=time.monotonic()
                if check_cancel and self.store.cancelled(body.user_id,turn['id']):
                    self.store.finish(body.user_id,turn['id'],'cancelled',answer=''.join(parts))
                    finished=True
                    yield 'cancelled',meta
                    return
                if event=='delta':parts.append(data['text'])
                if event=='reset':parts.clear()
                if event=='retrieval':meta.update(data)
                if event=='cancelled':
                    self.store.finish(body.user_id,turn['id'],'cancelled',answer=''.join(parts))
                    finished=True
                    yield 'cancelled',meta
                    return
                if event=='done':
                    if not body.deep_thinking:
                        data={**data,'analysis':{'mode':'normal','validation':'citations_only','searches':1,
                                               'elapsed_seconds':round(time.monotonic()-turn.get('_started',time.monotonic()),3)}}
                    self.store.save_analysis(body.user_id,turn['id'],data.get('analysis',{}))
                    status=self.store.finish(body.user_id,turn['id'],'completed',answer=data['answer'],sources=data['sources'])
                    finished=True
                    if status=='cancelled':
                        yield 'cancelled',meta
                        return
                    yield event,{**meta,**data}
                else:yield event,data
        except ServiceError as exc:
            self.store.finish(body.user_id,turn['id'],'failed',answer=''.join(parts),error=str(exc))
            finished=True
            raise
        finally:
            try:
                if stream is not None:stream.close()
                if not finished:self.store.finish(body.user_id,turn['id'],'cancelled',answer=''.join(parts))
            finally:self.slots.release()


def service(request):return request.app.state.chat_service


@router.get('/users')
def users(request:Request):return service(request).store.users()


@router.get('/documents/{document_id}/text')
def document_text(document_id:int,user_id:int,request:Request,offset:int=Query(default=0,ge=0)):
    return service(request).store.document_text(user_id,document_id,offset)


@router.get('/config')
def config(request:Request):
    cfg=request.app.state.settings
    return {'llm_provider':cfg.llm_provider,'llm_model':cfg.llm_model,
            'max_concurrent_generations':cfg.max_concurrent_generations,
            'memory_enabled':cfg.memory_enabled,'memory_max_turns':cfg.memory_max_turns,
            'memory_summary_enabled':cfg.memory_summary_enabled,
            'summary_model':cfg.summary_llm_model or cfg.llm_model,
            'deep_analysis':{'max_steps':cfg.deep_max_steps,'max_searches':cfg.deep_max_searches,
                             'retry_limit':cfg.deep_retry_limit,'timeout_seconds':cfg.deep_timeout_seconds,
                             'model':cfg.deep_llm_model or cfg.llm_model}}


@router.get('/users/{user_id}/conversations')
def conversations(user_id:int,request:Request):return service(request).store.conversations(user_id)


@router.post('/conversations',status_code=201)
def create(body:NewConversation,request:Request):return service(request).store.create(body.user_id,body.title)


@router.get('/conversations/{conversation_id}/messages')
def history(conversation_id:UUID,user_id:int,request:Request):
    return service(request).store.history(user_id,conversation_id)


@router.post('/conversations/{conversation_id}/cancel')
def cancel(conversation_id:UUID,body:CancelRequest,request:Request):
    return service(request).store.cancel(body.user_id,conversation_id,body.request_id)


@router.post('/conversations/{conversation_id}/messages')
def message(conversation_id:UUID,body:MessageRequest,request:Request):
    svc=service(request)
    turn,chunks,replay=svc.prepare(conversation_id,body)
    result=None
    for event,data in svc.run(body,turn,chunks,replay):
        if event=='done':result=data
        if event=='cancelled':raise HTTPException(409,'Generation was cancelled')
    return result


def next_event(iterator):
    try:return next(iterator)
    except StopIteration:return None


@router.post('/conversations/{conversation_id}/messages/stream',response_class=StreamingResponse,
             responses={200:{'content':{'text/event-stream':{}}}})
def message_stream(conversation_id:UUID,body:MessageRequest,request:Request):
    svc=service(request)
    turn,chunks,replay=svc.prepare(conversation_id,body)
    iterator=svc.run(body,turn,chunks,replay)

    async def events():
        try:
            while True:
                item=await anyio.to_thread.run_sync(next_event,iterator)
                if item is None:break
                event,data=item
                yield f'event: {event}\ndata: {json.dumps(data,ensure_ascii=False)}\n\n'
        except ServiceError as exc:
            yield f'event: error\ndata: {json.dumps({"detail":str(exc),"status":exc.status_code})}\n\n'
        except Exception:
            log.exception('Chat streaming failed')
            yield 'event: error\ndata: {"detail":"Chat streaming failed; check backend logs","status":500}\n\n'
        finally:
            # Ensure cancellation also closes upstream HTTP and releases the per-process slot.
            with anyio.CancelScope(shield=True):
                await anyio.to_thread.run_sync(iterator.close)

    return StreamingResponse(events(),media_type='text/event-stream',headers={
        'Cache-Control':'no-cache','X-Accel-Buffering':'no',
    })
