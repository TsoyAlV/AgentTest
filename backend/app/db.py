import psycopg
import time
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool, PoolTimeout

from .errors import ServiceError
from .schemas import Chunk


def configure_connection(conn):
    register_vector(conn)


class Retriever:
    def __init__(self, settings):
        self.pool = None
        dsn = settings.database_url.get_secret_value()
        if dsn:
            self.pool = ConnectionPool(
                conninfo=dsn, min_size=0, max_size=4, timeout=5,
                kwargs={"autocommit": True, "row_factory": dict_row, "connect_timeout": 5},
                configure=configure_connection, open=False,
            )

    def open(self):
        if self.pool:
            self.pool.open()

    def close(self):
        if self.pool:
            self.pool.close()

    def search(self, user_id, query, vector, limit, deadline=None):
        if self.pool is None:
            raise ServiceError("DATABASE_URL is not configured")
        try:
            with self.pool.connection() as conn:
                with conn.transaction():
                    # Fail closed if an administrator connection was configured accidentally.
                    role = conn.execute("""
                        SELECT rolsuper, rolbypassrls FROM pg_roles
                        WHERE rolname = current_user
                    """).fetchone()
                    if role["rolsuper"] or role["rolbypassrls"]:
                        raise ServiceError("Use rag_reader without SUPERUSER or BYPASSRLS")
                    owner = conn.execute("""
                        SELECT EXISTS (
                            SELECT 1 FROM pg_class c
                            JOIN pg_namespace n ON n.oid = c.relnamespace
                            WHERE n.nspname = 'public'
                              AND c.relname IN ('chunks', 'documents')
                              AND (NOT c.relrowsecurity OR pg_has_role(c.relowner, 'USAGE'))
                        ) AS unsafe
                    """).fetchone()
                    if owner["unsafe"]:
                        raise ServiceError("Enable RLS and use a non-owner database role")
                    if deadline is None:conn.execute("SET LOCAL statement_timeout = '15s'")
                    else:
                        remaining=deadline-time.monotonic()
                        if remaining<=0:raise ServiceError('Request time budget exhausted',504)
                        conn.execute("SELECT set_config('statement_timeout',%s,true)",(str(max(1,min(15000,int(remaining*1000)))),))
                    conn.execute("SELECT set_config('app.user_id', %s, true)", (str(user_id),))
                    conn.execute("SET LOCAL hnsw.ef_search = 100")
                    conn.execute("SET LOCAL hnsw.iterative_scan = 'strict_order'")
                    rows = conn.execute(
                        "SELECT * FROM public.hybrid_search(%s, %s, %s)",
                        (query, vector, limit),
                    ).fetchall()
            return [Chunk.model_validate(row) for row in rows]
        except (psycopg.Error, PoolTimeout):
            raise ServiceError("Database search failed; check connectivity, grants and migrations") from None
