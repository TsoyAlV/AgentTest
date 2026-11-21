BEGIN;
ALTER TABLE public.rag_turns ADD COLUMN IF NOT EXISTS retrieval_query text;
ALTER TABLE public.rag_turns ADD COLUMN IF NOT EXISTS memory_turn_ids uuid[] NOT NULL DEFAULT '{}';
ALTER TABLE public.rag_turns ADD COLUMN IF NOT EXISTS memory_status text NOT NULL DEFAULT 'legacy';
ALTER TABLE public.rag_turns ADD COLUMN IF NOT EXISTS memory_warning text;
COMMIT;
