from contextlib import contextmanager
from urllib.parse import quote

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from pgvector.psycopg import register_vector


class IngestionStore:
    def __init__(self,dsn,collection):
        if not dsn:raise ValueError('Set INGEST_DATABASE_URL or use --database-user and --prompt-password with an existing writer/admin login')
        self.dsn=dsn;self.collection=collection

    @contextmanager
    def connect(self):
        with psycopg.connect(self.dsn,autocommit=True,row_factory=dict_row,connect_timeout=10) as conn:
            register_vector(conn)
            yield conn

    @contextmanager
    def run_lock(self):
        with self.connect() as c:
            locked=c.execute("SELECT pg_try_advisory_lock(hashtextextended(%s,0)) AS locked",('ingest:'+self.collection,)).fetchone()['locked']
            if not locked:raise ValueError('An importer for this collection is already running')
            try:yield
            finally:c.execute("SELECT pg_advisory_unlock(hashtextextended(%s,0))",('ingest:'+self.collection,))

    def snapshot(self):
        with self.connect() as c:
            return {r['relative_path']:r for r in c.execute('SELECT * FROM public.file_sources WHERE collection=%s',(self.collection,)).fetchall()}

    def roles(self,names,create=False):
        with self.connect() as c,c.transaction():
            if create:
                for name in sorted(names):c.execute('INSERT INTO public.app_roles(name) VALUES(%s) ON CONFLICT(name) DO NOTHING',(name,))
            rows=c.execute('SELECT id,name FROM public.app_roles WHERE name=ANY(%s::text[])',(list(names),)).fetchall()
        mapping={r['name']:r['id'] for r in rows}
        missing=set(names)-mapping.keys()
        if missing:raise ValueError('Missing roles: '+', '.join(sorted(missing))+'. Use --create-roles after checking folder mapping.')
        return mapping

    def reusable(self,digest,signature):
        with self.connect() as c:
            return c.execute('''SELECT document_id FROM public.file_sources
                WHERE collection=%s AND indexed_hash=%s AND pipeline_signature=%s AND document_id IS NOT NULL
                ORDER BY id LIMIT 1''',(self.collection,digest,signature)).fetchone()

    @staticmethod
    def set_acl(c,document_id,roles):
        c.execute('DELETE FROM public.document_roles WHERE document_id=%s',(document_id,))
        for role in roles:c.execute('INSERT INTO public.document_roles(document_id,role_id) VALUES(%s,%s) ON CONFLICT DO NOTHING',(document_id,role))

    def register(self,entry,role_ids,status='processing'):
        with self.connect() as c,c.transaction():
            row=c.execute('''INSERT INTO public.file_sources(collection,relative_path,file_hash,role_names,status,file_size)
                VALUES(%s,%s,%s,%s,%s,%s) ON CONFLICT(collection,relative_path) DO UPDATE SET
                file_hash=excluded.file_hash,role_names=excluded.role_names,status=excluded.status,
                file_size=excluded.file_size,last_seen_at=now(),last_error=NULL RETURNING *''',
                (self.collection,entry.relative_path,entry.digest,entry.roles,status,entry.size)).fetchone()
            if row['document_id']:self.set_acl(c,row['document_id'],role_ids)
            return row

    def touch(self,entry,role_ids):
        row=self.register(entry,role_ids,status='indexed')
        return row['document_id'],row['chunk_count']

    def failed(self,source_id,error):
        with self.connect() as c:
            c.execute("UPDATE public.file_sources SET status='failed',last_error=%s WHERE id=%s",(error[:1000],source_id))

    def unavailable(self,paths,status):
        if not paths:return
        with self.connect() as c,c.transaction():
            rows=c.execute('''UPDATE public.file_sources SET status=%s WHERE collection=%s
                AND relative_path=ANY(%s::text[]) RETURNING document_id''',(status,self.collection,list(paths))).fetchall()
            for row in rows:
                if row['document_id']:self.set_acl(c,row['document_id'],[])

    def publish(self,source,entry,signature,model,role_ids,chunks=None,vectors=None,copy_document=None):
        with self.connect() as c,c.transaction():
            current=c.execute('SELECT * FROM public.file_sources WHERE id=%s FOR UPDATE',(source['id'],)).fetchone()
            if current['file_hash']!=entry.digest:raise ValueError('File was changed by another importer')
            doc=current['document_id']
            metadata={'source_type':'local_file','collection':self.collection,'relative_path':entry.relative_path,
                      'file_hash':entry.digest,'pipeline_signature':signature}
            if doc is None:
                uri='local://'+quote(self.collection,safe='')+'/'+quote(entry.relative_path,safe='/')
                doc=c.execute('INSERT INTO public.documents(url,title,content_hash,metadata) VALUES(%s,%s,%s,%s) RETURNING id',
                              (uri,entry.path.name,entry.digest,Jsonb(metadata))).fetchone()['id']
            c.execute('UPDATE public.documents SET url=%s,title=%s,content_hash=%s,metadata=%s,fetched_at=now() WHERE id=%s',
                      (f'/dev/documents/{doc}/text',entry.path.name,entry.digest,Jsonb(metadata),doc))
            c.execute('DELETE FROM public.chunks WHERE document_id=%s',(doc,))
            if copy_document is not None:
                if copy_document==doc:raise ValueError('Cannot copy a document onto itself')
                c.execute('''INSERT INTO public.chunks(document_id,chunk_no,section_title,content,embedding,embedding_model,metadata)
                    SELECT %s,chunk_no,section_title,content,embedding,embedding_model,
                           metadata || jsonb_build_object('relative_path',%s::text)
                    FROM public.chunks WHERE document_id=%s''',(doc,entry.relative_path,copy_document))
            else:
                with c.cursor() as cursor:
                    cursor.executemany('''INSERT INTO public.chunks(document_id,chunk_no,section_title,content,embedding,embedding_model,metadata)
                        VALUES(%s,%s,%s,%s,%s,%s,%s)''',[
                        (doc,i,chunk.section_title,chunk.content,vector,model,
                         Jsonb({**chunk.metadata,'relative_path':entry.relative_path}))
                        for i,(chunk,vector) in enumerate(zip(chunks,vectors,strict=True))
                    ])
            count=c.execute('SELECT count(*) AS n FROM public.chunks WHERE document_id=%s',(doc,)).fetchone()['n']
            if count==0:raise ValueError('Refusing to publish an empty document')
            self.set_acl(c,doc,role_ids)
            c.execute('''UPDATE public.file_sources SET document_id=%s,indexed_hash=%s,pipeline_signature=%s,
                status='indexed',chunk_count=%s,indexed_at=now(),last_error=NULL WHERE id=%s''',
                (doc,entry.digest,signature,count,source['id']))
            return doc,count
