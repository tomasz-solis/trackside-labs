-- Append-only record of issued forecasts (plan row 21). Run in the Supabase SQL Editor,
-- then run queries/forecast_record_append_only_check.sql.
--
-- One row per forecast as served: identity, the ordered rows exactly as served (core),
-- the information cutoff, the issue time and what produced it. Results never go here.
-- Rows are never changed: triggers refuse UPDATE, DELETE and TRUNCATE for every role,
-- service_role included (it bypasses row-level security, not triggers). A forecast
-- issued after its information cutoff is refused by the check constraint.

create table if not exists public.forecast_record (
    id                    bigserial   primary key,
    season                integer     not null,
    event                 text        not null,
    target                text        not null,  -- main_qualifying, grand_prix_race, sprint_qualifying, sprint_race
    checkpoint            text        not null,  -- PRE, FP1, FP2, FP3, SQ, Sprint, Q
    weather               text        not null,  -- dry, mixed, rain
    issued_at             timestamptz not null default now(),  -- set by the database
    information_cutoff_at timestamptz not null,
    boundary_signature    text        not null,  -- sessions elapsed when it was computed
    model_version         text        not null,
    config_hash           text        not null,
    learning_state_hash   text        not null,
    artifact_hash         text        not null,
    core                  jsonb       not null,
    content_hash          text        not null unique,  -- sha256 of core; a repeat insert does nothing
    replaces              bigint      null references public.forecast_record (id),
    replace_reason        text        null,
    constraint forecast_record_issued_before_cutoff check (issued_at <= information_cutoff_at),
    constraint forecast_record_replace_has_reason check ((replaces is null) = (replace_reason is null))
);

create index if not exists forecast_record_identity_idx
    on public.forecast_record (season, event, target, checkpoint, weather, id desc);

create or replace function public.forecast_record_stamp_issued_at()
returns trigger language plpgsql as $$
begin
    new.issued_at := now();
    return new;
end;
$$;

create or replace function public.forecast_record_refuse_change()
returns trigger language plpgsql as $$
begin
    raise exception 'forecast_record is append-only: % refused', tg_op;
end;
$$;

drop trigger if exists forecast_record_stamp on public.forecast_record;
create trigger forecast_record_stamp
    before insert on public.forecast_record
    for each row execute function public.forecast_record_stamp_issued_at();

drop trigger if exists forecast_record_no_change on public.forecast_record;
create trigger forecast_record_no_change
    before update or delete on public.forecast_record
    for each row execute function public.forecast_record_refuse_change();

drop trigger if exists forecast_record_no_truncate on public.forecast_record;
create trigger forecast_record_no_truncate
    before truncate on public.forecast_record
    for each statement execute function public.forecast_record_refuse_change();

alter table public.forecast_record enable row level security;
alter table public.forecast_record force row level security;

revoke all on table public.forecast_record from public, anon, authenticated;
revoke all on sequence public.forecast_record_id_seq from public, anon, authenticated;
grant select, insert on table public.forecast_record to service_role;
grant usage on sequence public.forecast_record_id_seq to service_role;

drop policy if exists "service_role_inserts_forecast_record" on public.forecast_record;
create policy "service_role_inserts_forecast_record"
    on public.forecast_record for insert
    to service_role
    with check (true);

drop policy if exists "service_role_reads_forecast_record" on public.forecast_record;
create policy "service_role_reads_forecast_record"
    on public.forecast_record for select
    to service_role
    using (true);
