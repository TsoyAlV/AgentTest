BEGIN;
ALTER TABLE public.rag_turns ADD COLUMN IF NOT EXISTS deep_thinking boolean NOT NULL DEFAULT false;
ALTER TABLE public.rag_turns ADD COLUMN IF NOT EXISTS analysis jsonb NOT NULL DEFAULT '{}';
COMMIT;
