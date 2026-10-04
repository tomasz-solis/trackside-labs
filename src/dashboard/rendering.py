"""Convenience exports for dashboard prediction rendering helpers."""

import pandas as pd

from src.dashboard import rendering_html, rendering_qualifying, rendering_race
from src.predictors.baseline.race.result_processing import assign_likely_range
from src.utils import config_loader

render_notice_banner = rendering_html.render_notice_banner
render_page_hero_deck = rendering_html.render_page_hero_deck
render_prediction_hero_deck = rendering_html.render_prediction_hero_deck


def _attach_likely_range(df: pd.DataFrame, result: dict, *, is_race: bool) -> pd.DataFrame:
    """Set ``likely_lo``/``likely_hi`` from the shown position and the current config.

    Recomputed at display time so a payload saved before the band existed, or with
    an older band, still shows the current calibrated range around its shown place.
    """
    if not is_race:
        table_name = "qualifying"
    elif str(result.get("starting_session_name", "")).strip().upper() == "SQ":
        table_name = "sprint"
    else:
        table_name = "race"
    rows = df[["position"]].to_dict("records")
    assign_likely_range(
        finish_order=rows, field_size=len(rows), cfg=config_loader, table_name=table_name
    )
    if rows and "likely_lo" in rows[0]:
        df["likely_lo"] = [row["likely_lo"] for row in rows]
        df["likely_hi"] = [row["likely_hi"] for row in rows]
    return df


def display_prediction_result(result: dict, prediction_name: str, is_race: bool = False) -> None:
    """Display a single prediction result (qualifying or race)."""
    results_key = "finish_order" if is_race else "grid"
    df = pd.DataFrame(result[results_key])
    df["position"] = df["position"].astype(int)
    df.attrs["input_confidence"] = result.get("input_confidence")
    result_mode = str(result.get("result_mode", "")).strip().upper()
    if result_mode != "ACTUAL" and not df.empty:
        df = _attach_likely_range(df, result, is_race=is_race)
    rendering_html.render_surface_header(
        title=prediction_name,
        summary=rendering_html._prediction_section_summary(result, is_race=is_race),
        eyebrow="Race projection" if is_race else "Qualifying projection",
    )

    highlight_cards = rendering_html._build_prediction_highlight_cards(df, result, is_race=is_race)
    rendering_html.render_stat_cards(highlight_cards)

    if result_mode == "ACTUAL":
        classification_note = str(result.get("classification_note", "")).strip()
        classification_caption = str(result.get("classification_caption", "")).strip()
        if classification_note:
            render_notice_banner(classification_note, tone="success", label="Completed session")
        if is_race:
            rendering_race._render_grid_penalty_notice(result)
            rendering_race._render_position_change_chart(
                df,
                result=result,
                prediction_name=prediction_name,
            )
        rendering_qualifying._render_actual_classification(
            df,
            caption=classification_caption
            or "Official result of the finished session, from FastF1.",
        )
        return

    qualifying_warning_messages: list[str] = []

    if not is_race:
        data_source = result.get("data_source", "Unknown")
        blend_used = result.get("blend_used", False)
        fp_blend_weight_used = result.get("fp_blend_weight_used")

        if blend_used:
            source_label = "practice data"
            if isinstance(data_source, str) and "checkpoint profile blend" in data_source.lower():
                source_label = "stored checkpoint snapshot"
            if isinstance(fp_blend_weight_used, int | float):
                practice_share = int(round(float(fp_blend_weight_used) * 100))
                model_share = max(0, 100 - practice_share)
                render_notice_banner(
                    (
                        f"Data source: {data_source} "
                        f"({practice_share}% {source_label} + {model_share}% model)."
                    ),
                    tone="info",
                    label="Input mix",
                )
            else:
                render_notice_banner(
                    f"Data source: {data_source} (70% {source_label} + 30% model).",
                    tone="info",
                    label="Input mix",
                )
        else:
            render_notice_banner(f"Data source: {data_source}.", tone="info", label="Input mix")
            if isinstance(data_source, str) and "Model-only" in data_source:
                qualifying_warning_messages.append(
                    "Low confidence: no practice or testing data for this weekend yet, so"
                    " the grid can look too close to team order."
                )
            elif isinstance(data_source, str) and "Testing short-run profile blend" in data_source:
                qualifying_warning_messages.append(
                    "Medium confidence: using team pace from testing, no laps from this "
                    "weekend yet. Expect wider ranges."
                )
        confidence_col = (
            "order_confidence"
            if "order_confidence" in df.columns
            and pd.to_numeric(df["order_confidence"], errors="coerce").notna().any()
            else "confidence"
        )
        if confidence_col in df.columns and not df.empty:
            mean_qualifying_confidence = float(
                pd.to_numeric(df[confidence_col], errors="coerce").mean()
            )
            if mean_qualifying_confidence < 50.0:
                qualifying_warning_messages.append(
                    "Tightly-packed grid: mean order confidence is "
                    f"{mean_qualifying_confidence:.1f}% (avg chance a driver qualifies within one "
                    "place of the projected slot). This reflects how separable the field is, "
                    "not just how many weekends the model has learned."
                )
        else:
            mean_qualifying_confidence = None

        team_cluster_warning = rendering_html._build_team_clustering_warning(
            df,
            mean_confidence=mean_qualifying_confidence,
        )
        if team_cluster_warning:
            qualifying_warning_messages.append(team_cluster_warning)

    compound_strategies = result.get("compound_strategies", {})
    pit_lap_distribution = result.get("pit_lap_distribution", {})

    if not is_race:
        rendering_html._render_collapsible_warnings(
            qualifying_warning_messages,
            title="Qualifying warnings",
        )

    if is_race:
        rendering_race._render_grid_penalty_notice(result)
        rendering_race._render_position_change_chart(
            df,
            result=result,
            prediction_name=prediction_name,
        )

    context_cards = rendering_html._build_context_cards(result, is_race=is_race)
    rendering_html.render_stat_cards(context_cards)

    if compound_strategies and is_race:
        rendering_race._render_compound_strategies(compound_strategies)

    if pit_lap_distribution and is_race:
        rendering_race._render_pit_lap_distribution(pit_lap_distribution)

    if is_race:
        rendering_race._render_race_result(df)
    else:
        teammate_head_to_head = result.get("teammate_head_to_head")
        if isinstance(teammate_head_to_head, list):
            rendering_qualifying._render_teammate_head_to_head_probabilities(teammate_head_to_head)
        rendering_qualifying._render_qualifying_result(df)
