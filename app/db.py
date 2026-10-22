import json
import os

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

DDL = """
create table if not exists targets (
  id bigserial primary key,
  symbol text not null,
  indication text not null,
  ensembl_id text not null,
  approved_symbol text not null,
  approved_name text,
  disease_id text not null,
  disease_name text not null,
  aliases jsonb not null default '[]',
  created_at timestamptz not null default now(),
  unique (ensembl_id, disease_id)
);
create table if not exists snapshots (
  id bigserial primary key,
  target_id bigint not null references targets(id) on delete cascade,
  as_of date not null,
  simulated boolean not null default false,
  data jsonb not null,
  created_at timestamptz not null default now()
);
create index if not exists snapshots_target_idx on snapshots (target_id, id desc);
create table if not exists decisions (
  id bigserial primary key,
  target_id bigint not null references targets(id) on delete cascade,
  snapshot_id bigint references snapshots(id) on delete set null,
  run_id bigint,
  decision text not null check (decision in ('advance', 'kill')),
  reasons text[] not null default '{}',
  note text,
  agent_verdict text,
  key_facts jsonb not null default '{}',
  created_at timestamptz not null default now()
);
create table if not exists events (
  id bigserial primary key,
  target_id bigint references targets(id) on delete cascade,
  kind text not null,
  payload jsonb not null default '{}',
  created_at timestamptz not null default now()
);
create table if not exists runs (
  id bigserial primary key,
  target_id bigint not null references targets(id) on delete cascade,
  kind text not null,
  status text not null default 'running',
  steps jsonb not null default '[]',
  latency_ms integer,
  input_tokens integer not null default 0,
  output_tokens integer not null default 0,
  claims_removed integer,
  output jsonb,
  error text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
update runs set status = 'error', error = 'interrupted by restart' where status = 'running';
"""

pool: AsyncConnectionPool | None = None


async def open_pool() -> None:
    global pool
    url = os.environ.get("DATABASE_URL")
    if not url:
        return
    new_pool = AsyncConnectionPool(url, min_size=1, max_size=5, open=False,
                                   kwargs={"row_factory": dict_row, "autocommit": True})
    await new_pool.open(wait=True, timeout=20)
    async with new_pool.connection() as conn:
        await conn.execute(DDL)
    pool = new_pool


async def close_pool() -> None:
    if pool:
        await pool.close()


def j(value) -> Jsonb:
    return Jsonb(value, dumps=lambda v: json.dumps(v, default=str))


async def fetch(sql: str, *args) -> list[dict]:
    async with pool.connection() as conn:
        cur = await conn.execute(sql, args)
        return await cur.fetchall() if cur.description else []


async def fetchone(sql: str, *args) -> dict | None:
    rows = await fetch(sql, *args)
    return rows[0] if rows else None
