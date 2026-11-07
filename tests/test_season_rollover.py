"""A new season starts from the previous season's end state in the store, not a stale file."""

from src.systems.season_rollover import seed_season_from_previous


class _Store:
    """Dict-backed stand-in for ArtifactStore."""

    def __init__(self, artifacts: dict[str, dict]):
        self.artifacts = artifacts
        self.saves: list[str] = []

    def load_artifact(self, artifact_type, artifact_key):
        return self.artifacts.get(artifact_key)

    def save_artifact(self, artifact_type, artifact_key, data, version=None):
        self.artifacts[artifact_key] = data
        self.saves.append(artifact_key)


def _end_of_2026() -> dict[str, dict]:
    return {
        "2026::car_characteristics": {
            "year": 2026,
            "version": 147,
            "races_completed": 31,
            "applied_events": ["2026::Mexico City Grand Prix::R"],
            "teams": {
                "McLaren": {
                    "overall_performance": 0.85,
                    "uncertainty": 0.12,
                    "current_season_performance": [0.80, 0.83],
                    "races_completed": 2,
                }
            },
        },
        "2026::driver_characteristics": {
            "year": 2026,
            "applied_events": ["2026::Mexico City Grand Prix::R"],
            "drivers": {
                "VER": {
                    "name": "Max Verstappen",
                    "experience": {"tier": "elite", "years_of_experience": 11},
                    "racecraft": {"skill_score": 0.85, "overtaking_skill": 0.90},
                    "pace": {"quali_pace": 0.92, "race_pace": 0.88},
                    "dnf_risk": {"dnf_rate": 0.05},
                    "bayesian": {"season_year": 2026},
                }
            },
        },
    }


def test_seed_starts_2027_from_the_2026_end_state_as_preseason_priors():
    store = _Store(_end_of_2026())

    assert seed_season_from_previous(2027, store=store) == [
        "car_characteristics",
        "driver_characteristics",
    ]

    car = store.artifacts["2027::car_characteristics"]
    assert car["year"] == 2027
    assert car["races_completed"] == 0
    assert car["data_freshness"] == "BASELINE_PRESEASON"
    assert "applied_events" not in car
    mclaren = car["teams"]["McLaren"]
    assert mclaren["overall_performance"] == 0.85  # learned strength carries over
    assert mclaren["current_season_performance"] == []
    drivers = store.artifacts["2027::driver_characteristics"]
    assert drivers["carried_over_from"] == 2026
    assert drivers["drivers"]["VER"]["racecraft"]["skill_score"] == 0.85
    assert drivers["drivers"]["VER"]["bayesian"] == {}
    # The 2026 state itself is untouched.
    assert store.artifacts["2026::car_characteristics"]["races_completed"] == 31


def test_seed_leaves_a_season_that_already_has_state_alone():
    store = _Store(_end_of_2026())
    seed_season_from_previous(2027, store=store)
    store.saves.clear()

    assert seed_season_from_previous(2027, store=store) == []
    assert store.saves == []


def test_seed_does_nothing_without_a_previous_season():
    store = _Store({})

    assert seed_season_from_previous(2027, store=store) == []
    assert store.saves == []
