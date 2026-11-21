BEGIN;
ALTER TABLE public.rag_turns ADD COLUMN IF NOT EXISTS memory_summary text NOT NULL DEFAULT '';
ALTER TABLE public.rag_turns ADD COLUMN IF NOT EXISTS summary_status text NOT NULL DEFAULT 'disabled';
ALTER TABLE public.rag_turns ADD COLUMN IF NOT EXISTS summary_model text;
COMMIT;
