"""Static markdown/HTML content for dashboard pages."""

MODEL_INSIGHTS_MARKDOWN = """
### How the model works

One model makes both the qualifying and the race forecast.

**1. Team strength**
- Blends the pre-season baseline with this season's results.
- Moves trust to this season fast: 45% at race 1, 95% from race 4.

**2. Qualifying**
- Uses the weekend's practice pace, or pre-season testing runs if there is no practice yet.
- From race 5, adjusts each car for the track: top speed, corner speeds and tyre wear.
- Adds team and driver adjustments, then runs 300 simulations.
- Shows the projected grid and a likely range for each driver.

**3. Race**
- Starts from the actual grid once qualifying is done, the predicted grid before that.
- Simulates the race lap by lap: pace, tyre wear, strategy, overtaking and retirements.
- Sprints have less randomness and a stronger grid effect.
- Win and podium odds are how often a driver finishes first or in the top 3 across 300 simulations.
- Shows a likely range for each driver.

**4. Learning**
- Each saved forecast is scored against the result, and the model learns per-driver and teammate errors.
- It skips replays, duplicate runs and missing or partial results.

**5. Model changes**
- A new component must pass promotion gates before it is added.
- Every change is measured on a replay of the season against a seed-noise floor.

**6. What 2026 showed**
- The grid predicts the race better than simulated pace, so the race forecast leans on qualifying.
- Car traits by track help the qualifying forecast. For the race the gain was too small to measure.
- Trained ML models (ridge, gradient boosting) did not beat this blend.
"""

QUALIFYING_HYPERPARAMETERS_MARKDOWN = """
**Qualifying:**
- Team 60%, driver 40% to start
- Practice pace when available, testing runs otherwise
- With no session data, weights rebalance and teammate and experience limits apply
- Learned corrections apply once there is history
- Car traits by track: top speed, slow, medium and fast corners, tyre wear
- Output: projected grid and likely range
"""

RACE_HYPERPARAMETERS_MARKDOWN = """
**Race:**
- Pace weight starts at 40% and shifts with how hard the circuit is to pass on
- Grid influence depends on overtaking difficulty and starting position
- Driver skill is normalised with the grid and pace terms in each simulation
- Includes retirements, lap 1 incidents, strategy variation and safety cars
- Finish order by each driver's median simulated place
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
