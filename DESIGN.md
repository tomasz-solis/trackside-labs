---
name: Trackside Labs
description: The pit-wall instrument for independent F1 forecasting: a calm graphite chassis with one hot readout.
colors:
  heat: "#FF4D2D"
  heat-soft: "#FF7A62"
  graphite: "#0B0F14"
  panel: "#111826"
  panel-alt: "#0F1623"
  soft-light: "#E8EDF2"
  steel: "#8B949E"
  success: "#48BF91"
  warning: "#F5B74A"
  info: "#78A7FF"
  link: "#9DC6FF"
  ck-pre: "#9DC6FF"
  ck-fp1: "#3671C6"
  ck-fp2: "#F4A7A3"
  ck-fp3: "#FF4D2D"
  ck-sq: "#F5B74A"
  ck-sprint: "#48BF91"
  ck-q: "#C084FC"
  ck-r: "#E8EDF2"
typography:
  display:
    fontFamily: "Sora, 'IBM Plex Sans', sans-serif"
    fontSize: "clamp(1.45rem, 1.4rem + 0.8vw, 2.3rem)"
    fontWeight: 600
    lineHeight: 1.08
    letterSpacing: "normal"
  headline:
    fontFamily: "Sora, 'IBM Plex Sans', sans-serif"
    fontSize: "2.1rem"
    fontWeight: 650
    lineHeight: 1.15
    letterSpacing: "0.015em"
  title:
    fontFamily: "Sora, 'IBM Plex Sans', sans-serif"
    fontSize: "1.28rem"
    fontWeight: 600
    lineHeight: 1.15
    letterSpacing: "0.008em"
  body:
    fontFamily: "'IBM Plex Sans', 'Segoe UI', sans-serif"
    fontSize: "1.03rem"
    fontWeight: 400
    lineHeight: 1.6
    letterSpacing: "0.01em"
  label:
    fontFamily: "'IBM Plex Sans', 'Segoe UI', sans-serif"
    fontSize: "0.72rem"
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: "0.09em"
rounded:
  sm: "9px"
  md: "12px"
  lg: "16px"
  xl: "18px"
  pill: "999px"
spacing:
  xs: "0.4rem"
  sm: "0.65rem"
  md: "1rem"
  lg: "1.25rem"
  xl: "2.6rem"
components:
  button-primary:
    backgroundColor: "{colors.heat}"
    textColor: "#FFFFFF"
    rounded: "11px"
    padding: "0.55rem 1.05rem"
  button-primary-hover:
    backgroundColor: "{colors.heat-soft}"
    textColor: "#FFFFFF"
  nav-pill:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.soft-light}"
    rounded: "{rounded.sm}"
    padding: "0.42rem 0.88rem"
  nav-pill-active:
    backgroundColor: "{colors.heat}"
    textColor: "#FFFFFF"
    rounded: "{rounded.sm}"
    padding: "0.42rem 0.88rem"
  surface-card:
    backgroundColor: "{colors.panel-alt}"
    textColor: "{colors.soft-light}"
    rounded: "{rounded.xl}"
    padding: "1rem 1.05rem"
  stat-card:
    backgroundColor: "{colors.panel-alt}"
    textColor: "{colors.soft-light}"
    rounded: "{rounded.md}"
    padding: "0.95rem 1rem"
  input:
    backgroundColor: "{colors.panel-alt}"
    textColor: "{colors.soft-light}"
    rounded: "{rounded.md}"
    padding: "0.5rem 0.75rem"
---

# Design system: Trackside Labs

## 1. Overview

**North star: the pit wall.** The app looks like a race engineer's screen. A near-black graphite base (#0B0F14), information in layered translucent panels, and one hot colour, Brake Heat orange (#FF4D2D), only where a decision or live state sits. Data leads. Energy is earned.

Panels float over a background of soft radial glows (warm top left, cool top right) with deep, soft shadows and a blurred sticky header. Nothing is flat, nothing shouts.

It rejects the betting-site look, the generic SaaS dashboard, and official F1 or team branding.

Key traits:

- Graphite base, one accent used sparingly.
- Translucent panels over a lit background.
- Sora for headings, IBM Plex Sans for UI and text.
- A checkpoint colour scale (PRE to R) carries the evidence timeline through every chart.
- WCAG 2.1 AA is the floor. Legibility beats energy.

## 2. Colours

| Name | Hex | Use |
|---|---|---|
| Brake Heat | #FF4D2D | Primary actions, active nav, live state, section kickers, FP3 on charts |
| Heat Soft | #FF7A62 | Hover, gradient tops, soft accent borders |
| Graphite | #0B0F14 | Page background |
| Panel / Panel Alt | #111826 / #0F1623 | Cards, tables, inputs |
| Soft Light | #E8EDF2 | Main text (0.90 to 0.96 opacity) |
| Steel | #8B949E | Muted text; must still pass AA |
| Success | #48BF91 | Finished sessions, positive movement |
| Warning | #F5B74A | Cautions, sprint qualifying, slight edges |
| Info / Link | #78A7FF / #9DC6FF | Notices and links |

Rules:

- **One voice.** Brake Heat is the only warm colour and covers at most about 10% of a screen. Never a large fill or decoration.
- **Two floors.** Content sits on Panel surfaces, which sit on graphite. Never directly on graphite.
- **The data scale is fixed.** PRE #9DC6FF, FP1 #3671C6, FP2 #F4A7A3, FP3 #FF4D2D, SQ #F5B74A, SPRINT #48BF91, Q #C084FC, R #E8EDF2. Used only in charts. Never recolour a checkpoint or reuse these colours for UI.

## 3. Typography

Sora for statements, IBM Plex Sans for reading.

| Level | Font | Size | Use |
|---|---|---|---|
| Display | Sora 600 | clamp(1.45rem, 1.4rem + 0.8vw, 2.3rem), line height 1.08 | Page or section header; the only fluid size |
| Headline | Sora 650 | 2.1rem | Main page title |
| Title | Sora 600 | 1rem to 1.28rem | Card titles, stat values, session tiles |
| Body | Plex 400 | 1.03rem, line height 1.6 | Text, capped at about 62 to 68 characters |
| Label | Plex 700 | 0.72rem, uppercase, spacing 0.08 to 0.12em | Kickers and stat labels |

Sora never sets body text or tables. Outside Display, sizes are fixed rem.

## 4. Elevation

Translucent panels (alpha 0.7 to 0.9) over a lit background, soft dark shadows, a 10px blur on the sticky header.

| Shadow | Value | Use |
|---|---|---|
| Card | `0 12px 28px rgba(0,0,0,0.26)` | Cards and panels |
| Header | `0 18px 36px rgba(0,0,0,0.28)` | Section headers |
| Table | `0 14px 34px rgba(0,0,0,0.34)` | Data tables |
| Heat glow | `0 10px 24px rgba(255,77,45,0.22)` | Primary buttons and accent elements only |

Shadows are large, soft and black (24 to 36px blur, 0.2 to 0.34 alpha). No tight drop shadows. The heat glow is the only coloured shadow.

## 5. Components

**Buttons.** 11px radius. Primary is a Heat Soft to Heat gradient, white text, padding 0.55rem 1.05rem, heat glow. Hover brightens and deepens the glow to rgba(255,77,45,0.30). Focus shows a 2px Brake Heat outline at 0.6 alpha, offset 2px.

**Cards.** 16 to 18px radius on large surfaces, 12px on stat and session tiles. Translucent Panel gradient, never a flat fill. Hairline border `1px solid rgba(232,237,242,0.10 to 0.16)`. Padding about 1rem, 1.2rem on section headers, which also get a faint heat glow top right.

**Inputs.** Panel Alt gradient, 12 to 14px radius, hairline border at 0.14 to 0.18 alpha. Values at 0.94 alpha, placeholders at 0.45. Every field has a visible focus state.

**Navigation.** A pill group in a dark trough. Active pill uses the heat gradient with white text. The sidebar marks the active item with a left heat bar. On mobile the top nav scrolls and tap targets are at least 2.35rem tall.

**Charts.** Plotly on a transparent background inside a 16px-radius panel, max width about 980px. Grid lines Soft Light at 0.08 to 0.18 alpha, text Soft Light at 13 to 14px. Series colours come only from the checkpoint scale.

**Matchup card.** Team head to head with a bar that fills left or right from the middle. Colour shows the size of the edge: steel (even), amber (slight), info blue (moderate), Brake Heat (clear).

## 6. Do and don't

Do:

- Keep Brake Heat rare: primary action, current selection, live state.
- Put content on Panel surfaces, never on bare graphite.
- Use Sora for statements and Plex for anything dense.
- Use large soft shadows; keep the heat glow for interactive accents.
- Use the checkpoint scale only for data.
- Hold AA contrast, visible focus and reduced motion.

Don't:

- Look like a betting site, a SaaS template or official F1 branding.
- Use the heat colour as a fill, gradient text or neutral shadow.
- Use tight, hard drop shadows.
- Set body text or tables in Sora, or let muted text drop below AA.
