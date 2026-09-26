"""Static markdown/HTML content for dashboard pages."""

MODEL_INSIGHTS_MARKDOWN = """
### How the model works

One predictor (`Baseline2026Predictor`) makes both the qualifying and the race forecast.

**1. Team strength**
- Blends the pre-season baseline with this season's results.
- Moves trust to this season fast: 45% at race 1, 95% from race 4.

**2. Qualifying**
- Uses the weekend's practice pace, or pre-season testing runs if there is no practice yet.
- Adds team and driver adjustments, then runs Monte Carlo simulations.
- Shows the median grid position and a confidence range.

**3. Race**
- Starts from the actual grid once qualifying is done, the predicted grid before that.
- Simulates the race lap by lap: pace, tyre wear, strategy, overtaking and retirements.
- Sprints have less randomness and a stronger grid effect.
- Podium odds come from how often each driver finishes top 3 in the simulations.

**4. Learning**
- Each saved forecast is scored against the result, and the model learns per-driver and teammate errors.
- It skips replays, duplicate runs and missing or partial results.

**5. Model changes**
- A new component must pass promotion gates before it is added.
- Every change is measured on a replay of the season against a seed-noise floor.
"""

QUALIFYING_HYPERPARAMETERS_MARKDOWN = """
**Qualifying:**
- Team 60%, driver 40% to start
- Practice pace when available, testing runs otherwise
- With no session data, weights rebalance and teammate and experience limits apply
- Learned corrections apply once there is history
- Output: median grid position and confidence range
"""

RACE_HYPERPARAMETERS_MARKDOWN = """
**Race:**
- Pace weight starts at 40% and shifts with how hard the circuit is to pass on
- Grid influence depends on overtaking difficulty and starting position
- Driver skill is normalised with the grid and pace terms in each simulation
- Includes retirements, lap 1 incidents, strategy variation and safety cars
- Podium odds from simulated finishing order, smoothed so they never rise down the grid
"""

CONTACT_PAGE_HTML = """
<div class="contact-grid">
  <section class="contact-card">
    <h3>Links</h3>
    <div class="contact-link-stack">
      <a class="contact-link-row" href="https://github.com/tomasz-solis/trackside-labs" target="_blank" rel="noopener noreferrer">
        <span class="contact-link-row__label">GitHub</span>
        <span class="contact-link-row__value">trackside-labs</span>
      </a>
      <a class="contact-link-row" href="https://www.linkedin.com/in/tomaszsolis/" target="_blank" rel="noopener noreferrer">
        <span class="contact-link-row__label">LinkedIn</span>
        <span class="contact-link-row__value">Tomasz Solis</span>
      </a>
    </div>
  </section>
  <section class="contact-card">
    <h3>What it does</h3>
    <p>Forecasts every 2026 race weekend, learns from each result and tracks its own accuracy.</p>
    <ul>
      <li>Team strength from the baseline, testing and this season</li>
      <li>Qualifying and race simulations that use practice data</li>
      <li>Forecasts saved every session and scored after the race</li>
    </ul>
  </section>
</div>
<section class="contact-card contact-card--full">
  <h3>Disclaimer</h3>
  <p>Independent analytics project. Not affiliated with any racing series, team, or governing body.</p>
</section>
"""
