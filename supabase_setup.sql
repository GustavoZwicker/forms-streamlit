-- ============================================================================
-- Setup para a pesquisa de deepfakes.
-- Cole tudo isto no Supabase: Dashboard -> SQL Editor -> New query -> Run.
-- Roda uma vez. Pode rodar de novo sem problema (é idempotente).
-- ============================================================================

-- 1) Contagem por grupo (semeia os 3 grupos com pesos iguais 1:1:1).
create table if not exists counts (
  grp    text primary key,
  n      integer not null default 0,
  weight numeric not null default 1
);
insert into counts (grp, weight) values
  ('Controle', 1), ('Checklist', 1), ('XAI', 1)
on conflict (grp) do nothing;

-- 2) Respostas dos participantes (uma linha por participante).
create table if not exists responses (
  id                 bigint generated always as identity primary key,
  created_at         timestamptz not null default now(),
  "timestamp"        text,
  participant_id     text,
  grp                text,
  consented          text,
  age_range          text,
  gender             text,
  education          text,
  ai_familiarity     text,
  deepfake_knowledge text,
  social_media_freq  text,
  used_ai            text,
  check_2_1          text,
  check_2_2          text,
  check_2_3          text,
  check_score        text,
  task_json          text,
  task_score         text,
  final_json         text,
  complete           text
);

-- 3) Estímulos: os vídeos + metadados (probabilidade e explicação XAI).
--    'video' = nome do arquivo no bucket de Storage (ou uma URL completa).
--    'label' = real/deepfake (opcional; se preenchido em todos, calcula acertos).
--    prob_deepfake e explanation aparecem SÓ para o grupo XAI.
create table if not exists stimuli (
  sort_order    numeric,
  video         text,
  label         text,
  prob_deepfake text,
  explanation   text
);

-- 4) Atribuição balanceada, atômica e livre de corrida.
--    Um advisory lock serializa as atribuições; escolhe o grupo mais "atrasado"
--    em relação ao peso e incrementa, tudo numa transação.
create or replace function assign_group()
returns text
language plpgsql
as $$
declare
  chosen text;
begin
  perform pg_advisory_xact_lock(987654321);
  select grp into chosen
    from counts
    order by (n + 1)::numeric / weight asc, random()
    limit 1;
  update counts set n = n + 1 where grp = chosen;
  return chosen;
end;
$$;

-- ============================================================================
-- Observações:
-- * O app usa a chave 'service_role' (server-side, nos secrets do Streamlit),
--   que ignora RLS. Por isso as tabelas podem ficar com RLS ligado (padrão) e
--   nenhuma policy: ninguém sem a chave acessa os dados. NÃO exponha essa chave.
-- * Para mudar a proporção de um grupo, ajuste 'weight' na tabela counts.
--   Ex.: update counts set weight = 2 where grp = 'XAI';
-- * Para zerar as contagens em um teste: update counts set n = 0;
-- ============================================================================
