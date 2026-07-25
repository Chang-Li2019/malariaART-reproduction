"""Load config.yaml into a set of resolved paths and analysis parameters.

Every script calls load_config() and reads absolute paths off the returned
Config, so no path is hard-coded anywhere else in the package.
"""

from pathlib import Path

import yaml


class Config:
    """Absolute input/output paths and analysis parameters, resolved from config.yaml."""

    def __init__(self, config_path: Path):
        document = yaml.safe_load(config_path.read_text())
        self.release_root = config_path.parent
        self._resolve_paths(document, config_path.parent)
        self._read_params(document["params"])
        self.excluded_samples = tuple(document.get("excluded_samples", []))

    def _resolve_paths(self, document: dict, root: Path) -> None:
        """Resolve every input/output path against the release root."""
        paths = document["paths"]
        self.phenotype_trac1 = (root / paths["phenotype_trac1"]).resolve()
        self.phenotype_trac2 = (root / paths["phenotype_trac2"]).resolve()
        self.cohort_metadata = (root / paths["cohort_metadata"]).resolve()
        self.pf7_metadata = (root / paths["pf7_metadata"]).resolve()
        self.reference_proteins = (root / paths["reference_proteins"]).resolve()
        self.reference_cds = (root / paths["reference_cds"]).resolve()
        self.sequences_dir = (root / paths["sequences_dir"]).resolve()
        self.gene_list_plasmodb_173 = (root / paths["gene_list_plasmodb_173"]).resolve()
        self.gene_list_panel_167 = (root / paths["gene_list_panel_167"]).resolve()
        self.gene_list_kucharski_12 = (root / paths["gene_list_kucharski_12"]).resolve()
        self.gene_list_dms_56 = (root / paths["gene_list_dms_56"]).resolve()
        self.evidence_scores = (root / paths["evidence_scores"]).resolve()
        self.published_supplementary_csv = (root / paths["published_supplementary_csv"]).resolve()
        self.published_source_tables = (root / paths["published_source_tables"]).resolve()
        self.figure_source_out = (root / paths["figure_source_out"]).resolve()
        self.rf_models = (root / paths["rf_models"]).resolve()
        self.dms_results = (root / paths["dms_results"]).resolve()
        self.rankscore_results = (root / paths["rankscore_results"]).resolve()
        self.permutation = (root / paths["permutation"]).resolve()
        self.feature_cache = (root / document["external_feature_cache"]).resolve()

    def _read_params(self, params: dict) -> None:
        """Copy the analysis parameters onto the config."""
        self.clearance_threshold_hours = params["clearance_threshold_hours"]
        self.split_train_max_year = params["split_train_max_year"]
        self.split_valid_years = tuple(params["split_valid_years"])
        self.split_test_min_year = params["split_test_min_year"]
        self.rf_n_estimators = params["rf_n_estimators"]
        self.random_seed = params["random_seed"]
        self.n_permutations = params["n_permutations"]


def default_config_path() -> Path:
    """Path to config.yaml at the release root, one level above code/."""
    return Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(config_path: Path | None = None) -> Config:
    """Load config.yaml, using the release-root default when no path is given."""
    return Config(config_path if config_path is not None else default_config_path())
