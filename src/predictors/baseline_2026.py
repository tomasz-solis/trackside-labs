"""2026 Baseline Predictor - Monte Carlo qualifying/race simulation with weight-scheduled team strength."""

import logging
import os
from pathlib import Path

from src.persistence.artifact_store import ArtifactStore
from src.predictors.baseline import (
    BaselineDataMixin,
    BaselineQualifyingMixin,
    BaselineRaceMixin,
)
from src.utils.config_loader import Config
from src.utils.data_generator import ensure_baseline_exists

logger = logging.getLogger(__name__)


class Baseline2026Predictor(
    BaselineDataMixin,
    BaselineQualifyingMixin,
    BaselineRaceMixin,
):
    """Primary 2026 predictor with weight-scheduled team strength and Monte Carlo simulation."""

    def __init__(
        self,
        data_dir: str = "data/processed",
        seed: int = 42,
        config: Config | None = None,
        artifact_store: ArtifactStore | None = None,
    ):
        """Initialize predictor with optional injectable config/artifact store."""
        BaselineDataMixin.__init__(self)
        self.seed = seed

        data_dir_path = Path(data_dir)
        if not data_dir_path.is_absolute():
            env_data_dir = os.getenv("F1_DATA_DIR")
            if env_data_dir:
                self.data_dir = (
                    Path(env_data_dir) / data_dir
                    if data_dir != "data/processed"
                    else Path(env_data_dir)
                )
            else:
                self.data_dir = Path.cwd() / data_dir
        else:
            self.data_dir = data_dir_path

        logger.info("Ensuring baseline data is ready...")
        ensure_baseline_exists(self.data_dir)

        self.artifact_store = artifact_store or ArtifactStore(
            data_root=self.data_dir.parent if self.data_dir.name == "processed" else self.data_dir
        )
        self.config = config or Config()
        self.load_data()
