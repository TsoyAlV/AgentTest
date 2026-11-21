-- Apply once as database administrator to the existing rag database.
BEGIN;
CREATE TABLE IF NOT EXISTS public.rag_conversations (
    id uuid PRIMARY KEY,
    user_id bigint NOT NULL REFERENCES public.app_users(id),
    title text NOT NULL CHECK (length(title) BETWEEN 1 AND 120),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS rag_conversations_user_idx
ON public.rag_conversations(user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS public.rag_turns (
    id uuid PRIMARY KEY,
    conversation_id uuid NOT NULL REFERENCES public.rag_conversations(id) ON DELETE CASCADE,
    request_id uuid NOT NULL,
    client_id text NOT NULL CHECK (length(client_id) BETWEEN 1 AND 100),
    question text NOT NULL,
    answer text NOT NULL DEFAULT '',
    sources jsonb NOT NULL DEFAULT '[]',
    evidence_document_ids bigint[] NOT NULL DEFAULT '{}',
    status text NOT NULL CHECK (status IN ('generating','completed','cancelled','failed')),
    error text,
    cancel_requested boolean NOT NULL DEFAULT false,
    lease_until timestamptz NOT NULL DEFAULT (now() + interval '1 hour'),
    created_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    UNIQUE (conversation_id, request_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS rag_turns_one_active
ON public.rag_turns(conversation_id) WHERE status = 'generating';
CREATE INDEX IF NOT EXISTS rag_turns_history_idx ON public.rag_turns(conversation_id, created_at, id);

ALTER TABLE public.rag_conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.rag_turns ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rag_conversations_owner ON public.rag_conversations;
CREATE POLICY rag_conversations_owner ON public.rag_conversations FOR ALL TO rag_reader
USING (user_id = nullif(current_setting('app.user_id', true),'')::bigint)
WITH CHECK (user_id = nullif(current_setting('app.user_id', true),'')::bigint);
DROP POLICY IF EXISTS rag_turns_owner ON public.rag_turns;
CREATE POLICY rag_turns_owner ON public.rag_turns FOR ALL TO rag_reader
USING (EXISTS (SELECT 1 FROM public.rag_conversations c WHERE c.id=conversation_id))
WITH CHECK (EXISTS (SELECT 1 FROM public.rag_conversations c WHERE c.id=conversation_id));

GRANT SELECT (id, auth_subject, is_active) ON public.app_users TO rag_reader;
GRANT SELECT ON public.app_roles, public.user_roles TO rag_reader;
GRANT SELECT, INSERT, UPDATE ON public.rag_conversations, public.rag_turns TO rag_reader;
COMMIT;
