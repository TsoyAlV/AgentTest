from contextlib import contextmanager
from uuid import uuid4

import psycopg
from fastapi import HTTPException
from psycopg.types.json import Jsonb
from psycopg_pool import PoolTimeout

from .errors import ServiceError


class ChatStore:
    """Short transactions only. RLS user context never survives the transaction."""
    def __init__(self, retriever):
        self.retriever = retriever

    @contextmanager
    def transaction(self, user_id=None):
        pool = self.retriever.pool
        if pool is None:
            raise ServiceError("DATABASE_URL is not configured")
        try:
            with pool.connection() as conn, conn.transaction():
                conn.execute("SET LOCAL statement_timeout = '10s'")
                role = conn.execute("SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname=current_user").fetchone()
                if role['rolsuper'] or role['rolbypassrls']:
                    raise ServiceError("Use rag_reader without SUPERUSER or BYPASSRLS")
                unsafe = conn.execute("""SELECT EXISTS (
                    SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
                    WHERE n.nspname='public' AND c.relname IN ('rag_conversations','rag_turns')
                    AND (NOT c.relrowsecurity OR pg_has_role(c.relowner,'USAGE'))
                ) AS unsafe""").fetchone()['unsafe']
                if unsafe:
                    raise ServiceError("Chat tables require RLS and a non-owner database role")
                if user_id is not None:
                    conn.execute("SELECT set_config('app.user_id',%s,true)", (str(user_id),))
                    if not conn.execute("SELECT id FROM public.app_users WHERE id=%s AND is_active", (user_id,)).fetchone():
                        raise HTTPException(404, "Active user not found")
                yield conn
        except psycopg.errors.UniqueViolation:
            raise HTTPException(409, "This chat is generating a response; use another chat or retry request_id") from None
        except (psycopg.Error, PoolTimeout):
            raise ServiceError("Chat storage unavailable; apply backend migrations and check database grants") from None

    def users(self):
        with self.transaction() as c:
            return c.execute("""SELECT u.id,u.auth_subject,
                coalesce(array_agg(r.name ORDER BY r.name) FILTER (WHERE r.name IS NOT NULL),'{}') AS roles
                FROM public.app_users u LEFT JOIN public.user_roles ur ON ur.user_id=u.id
                LEFT JOIN public.app_roles r ON r.id=ur.role_id WHERE u.is_active
                GROUP BY u.id,u.auth_subject ORDER BY u.id""").fetchall()

    def conversations(self, user_id):
        with self.transaction(user_id) as c:
            return c.execute("SELECT * FROM public.rag_conversations ORDER BY updated_at DESC,id LIMIT 200").fetchall()

    def document_text(self,user_id,document_id,offset=0,limit=10):
        with self.transaction(user_id) as c:
            document=c.execute('SELECT id,title FROM public.documents WHERE id=%s',(document_id,)).fetchone()
            if not document:raise HTTPException(404,'Document not found for this user')
            rows=c.execute('''SELECT chunk_no,section_title,content,metadata FROM public.chunks
                WHERE document_id=%s ORDER BY chunk_no OFFSET %s LIMIT %s''',(document_id,offset,limit+1)).fetchall()
            return {**document,'chunks':rows[:limit],'next_offset':offset+limit if len(rows)>limit else None}

    def create(self, user_id, title):
        with self.transaction(user_id) as c:
            return c.execute("INSERT INTO public.rag_conversations(id,user_id,title) VALUES(%s,%s,%s) RETURNING *",
                             (uuid4(),user_id,title)).fetchone()

    @staticmethod
    def require_conversation(c, conversation_id):
        row = c.execute("SELECT * FROM public.rag_conversations WHERE id=%s", (conversation_id,)).fetchone()
        if not row:
            raise HTTPException(404,"Chat not found for this user")
        return row

    @staticmethod
    def visible(c, row):
        ids = row['evidence_document_ids']
        if ids and not c.execute("SELECT coalesce(bool_and(public.can_read_document(id)),false) AS allowed FROM unnest(%s::bigint[]) AS ids(id)", (ids,)).fetchone()['allowed']:
            return {**row, 'answer': 'Ответ скрыт: доступ к исходным документам изменился.', 'sources': [],
                    'retrieval_query':None,'memory_turn_ids':[],'memory_warning':None,'memory_summary':'',
                    'summary_status':'redacted','summary_model':None,'analysis':{},'redacted': True}
        return {**row,'redacted':False}

    def history(self, user_id, conversation_id):
        with self.transaction(user_id) as c:
            self.require_conversation(c,conversation_id)
            rows = c.execute("""SELECT * FROM (
                SELECT * FROM public.rag_turns WHERE conversation_id=%s ORDER BY created_at DESC,id DESC LIMIT 200
                ) t ORDER BY created_at,id""", (conversation_id,)).fetchall()
            return [self.visible(c,row) for row in rows]

    def memory_history(self,user_id,conversation_id,exclude_turn_id,max_turns):
        with self.transaction(user_id) as c:
            self.require_conversation(c,conversation_id)
            rows=c.execute("""SELECT * FROM (
                SELECT * FROM public.rag_turns WHERE conversation_id=%s
                  AND id<>%s AND status='completed'
                ORDER BY created_at DESC,id DESC LIMIT %s
                ) t ORDER BY created_at,id""",(conversation_id,exclude_turn_id,max_turns)).fetchall()
            return [self.visible(c,row) for row in rows]

    def save_memory(self,user_id,turn_id,memory):
        with self.transaction(user_id) as c:
            c.execute("""UPDATE public.rag_turns SET retrieval_query=%s,memory_turn_ids=%s,
                         memory_status=%s,memory_warning=%s,memory_summary=%s,summary_status=%s,
                         summary_model=%s WHERE id=%s""",
                      (memory.query,memory.turn_ids,memory.status,memory.warning,memory.summary,
                       memory.summary_status,memory.summary_model,turn_id))

    def save_analysis(self,user_id,turn_id,analysis):
        with self.transaction(user_id) as c:
            c.execute('UPDATE public.rag_turns SET analysis=%s WHERE id=%s',(Jsonb(analysis),turn_id))

    def begin(self, user_id, conversation_id, body):
        with self.transaction(user_id) as c:
            self.require_conversation(c,conversation_id)
            previous = c.execute("SELECT * FROM public.rag_turns WHERE conversation_id=%s AND request_id=%s",
                                 (conversation_id,body.request_id)).fetchone()
            if previous:
                if (previous['question'] != body.message or previous['client_id'] != body.client_id
                        or previous.get('deep_thinking',False)!=getattr(body,'deep_thinking',False)):
                    raise HTTPException(409,"request_id was already used for a different message")
                if previous['status'] != 'completed':
                    raise HTTPException(409,"Request already exists; inspect chat history or use a new request_id")
                return self.visible(c,previous), True
            # Recover a slot after a crashed worker. Normal completion/cancellation releases it immediately.
            c.execute("""UPDATE public.rag_turns SET status='failed',error='Worker lease expired',finished_at=now()
                         WHERE conversation_id=%s AND status='generating' AND lease_until<now()""", (conversation_id,))
            row = c.execute("""INSERT INTO public.rag_turns(id,conversation_id,request_id,client_id,question,deep_thinking,status)
                VALUES(%s,%s,%s,%s,%s,%s,'generating') RETURNING *""",
                (uuid4(),conversation_id,body.request_id,body.client_id,body.message,getattr(body,'deep_thinking',False))).fetchone()
            c.execute("UPDATE public.rag_conversations SET updated_at=now() WHERE id=%s", (conversation_id,))
            c.execute("""UPDATE public.rag_conversations SET title=%s
                         WHERE id=%s AND title='Новый диалог'""", (body.message[:80],conversation_id))
            return row, False

    def evidence(self,user_id,turn_id,document_ids):
        with self.transaction(user_id) as c:
            c.execute("UPDATE public.rag_turns SET evidence_document_ids=%s WHERE id=%s", (document_ids,turn_id))

    def cancelled(self,user_id,turn_id):
        with self.transaction(user_id) as c:
            row=c.execute("SELECT cancel_requested FROM public.rag_turns WHERE id=%s",(turn_id,)).fetchone()
            return not row or row['cancel_requested']

    def cancel(self,user_id,conversation_id,request_id):
        with self.transaction(user_id) as c:
            self.require_conversation(c,conversation_id)
            row=c.execute("""UPDATE public.rag_turns SET cancel_requested=true
                WHERE conversation_id=%s AND request_id=%s AND status='generating' RETURNING id""",
                (conversation_id,request_id)).fetchone()
            return {'cancel_requested':bool(row)}

    def finish(self,user_id,turn_id,status,answer='',sources=None,error=None):
        with self.transaction(user_id) as c:
            # A cancel request wins a race with completion.
            row=c.execute("""UPDATE public.rag_turns SET
                status=CASE WHEN cancel_requested THEN 'cancelled' ELSE %s END,
                answer=%s,sources=%s,error=%s,finished_at=now()
                WHERE id=%s AND status='generating' RETURNING status""",(status,answer,Jsonb(sources or []),error,turn_id)).fetchone()
            return row['status'] if row else None
