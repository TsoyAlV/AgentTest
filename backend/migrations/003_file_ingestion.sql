BEGIN;
CREATE TABLE IF NOT EXISTS public.file_sources (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    collection text NOT NULL,
    relative_path text NOT NULL,
    document_id bigint UNIQUE REFERENCES public.documents(id) ON DELETE SET NULL,
    file_hash text NOT NULL,
    indexed_hash text,
    pipeline_signature text,
    role_names text[] NOT NULL DEFAULT '{}',
    status text NOT NULL DEFAULT 'pending',
    file_size bigint NOT NULL DEFAULT 0,
    chunk_count integer NOT NULL DEFAULT 0,
    last_seen_at timestamptz NOT NULL DEFAULT now(),
    indexed_at timestamptz,
    last_error text,
    UNIQUE(collection,relative_path)
);
CREATE INDEX IF NOT EXISTS file_sources_reuse_idx
ON public.file_sources(collection,indexed_hash,pipeline_signature) WHERE document_id IS NOT NULL;
-- Backend reads indexed source metadata through the existing document RLS.
ALTER TABLE public.file_sources ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS file_sources_read ON public.file_sources;
CREATE POLICY file_sources_read ON public.file_sources FOR SELECT TO rag_reader
USING (document_id IS NOT NULL AND public.can_read_document(document_id));
GRANT SELECT ON public.file_sources TO rag_reader;
COMMIT;
