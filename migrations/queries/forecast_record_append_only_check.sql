-- Check that forecast_record refuses changes in the database itself (plan row 21, test 2).
-- Run in the Supabase SQL Editor after migration 007. It inserts one probe row inside a
-- transaction and rolls back, so nothing stays. It passes when it finishes without an
-- error (the notice says "forecast_record append-only check passed"); it fails loudly.

begin;

insert into public.forecast_record (
    season, event, target, checkpoint, weather, information_cutoff_at, boundary_signature,
    model_version, config_hash, learning_state_hash, artifact_hash, core, content_hash
) values (
    1900, 'append-only probe', 'main_qualifying', 'PRE', 'dry', now() + interval '1 day', 'probe',
    'probe', 'probe', 'probe', 'probe', '{}'::jsonb, 'append-only-probe'
);

do $$
declare
    refused int := 0;
begin
    begin
        update public.forecast_record set model_version = 'changed' where content_hash = 'append-only-probe';
    exception when raise_exception then refused := refused + 1;
    end;
    begin
        delete from public.forecast_record where content_hash = 'append-only-probe';
    exception when raise_exception then refused := refused + 1;
    end;
    begin
        truncate public.forecast_record;
    exception when raise_exception then refused := refused + 1;
    end;
    begin
        insert into public.forecast_record (
            season, event, target, checkpoint, weather, information_cutoff_at, boundary_signature,
            model_version, config_hash, learning_state_hash, artifact_hash, core, content_hash
        ) values (
            1900, 'append-only probe', 'main_qualifying', 'PRE', 'dry', now() - interval '1 second',
            'probe', 'probe', 'probe', 'probe', 'probe', '{}'::jsonb, 'after-cutoff-probe'
        );
    exception when check_violation then refused := refused + 1;
    end;
    if refused <> 4 then
        raise exception 'forecast_record append-only check FAILED: % of 4 changes refused', refused;
    end if;
    if (select count(*) from public.forecast_record where content_hash = 'append-only-probe') <> 1 then
        raise exception 'forecast_record append-only check FAILED: probe row changed';
    end if;
    raise notice 'forecast_record append-only check passed';
end;
$$;

rollback;
