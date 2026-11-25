-- Let a forecast that returns to an earlier content be recorded again (A, B, then A).
-- Run in the Supabase SQL Editor after 007, then rerun
-- queries/forecast_record_append_only_check.sql.
--
-- 007 made content_hash unique, so the returning A was refused and the record kept B as
-- the latest forecast while the site served A. content_hash stays the sha256 of core;
-- uniqueness moves to (content_hash, replaces): one issuance per content and predecessor.
-- Rows are not touched, so the append-only triggers do not fire.

begin;

do $$
declare
    unique_name text;
begin
    select con.conname into unique_name
    from pg_constraint con
    join pg_attribute att
        on att.attrelid = con.conrelid and att.attnum = any (con.conkey)
    where con.conrelid = 'public.forecast_record'::regclass
        and con.contype = 'u'
        and att.attname = 'content_hash'
        and array_length(con.conkey, 1) = 1;
    if unique_name is not null then
        execute format('alter table public.forecast_record drop constraint %I', unique_name);
    end if;
end;
$$;

create unique index if not exists forecast_record_first_issue_once
    on public.forecast_record (content_hash)
    where replaces is null;

create unique index if not exists forecast_record_issue_once_per_predecessor
    on public.forecast_record (content_hash, replaces)
    where replaces is not null;

commit;
