"""Preserve the pre-A-share synthetic smoke artifacts, without mixing them into real results."""

from market_microstructure.utils.paths import PROJECT_ROOT, project_path


def main() -> None:
    """Move only artifacts explicitly created by this research pipeline inside this workspace."""
    destination = project_path("outputs/engineering_validation")
    if destination.joinpath("README.md").exists():
        raise RuntimeError("One-time archive already exists; refusing to move current research results")
    destination.mkdir(parents=True, exist_ok=True)
    for directory in ("outputs/backtests", "outputs/models", "outputs/tables"):
        for source in project_path(directory).iterdir():
            if source.name.startswith(("ashare_", "public_", "reconstruction_")) or source.name == ".gitkeep":
                continue
            target = destination / source.parent.name / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_file() and source.resolve().is_relative_to(PROJECT_ROOT) and not target.exists():
                source.rename(target)
    destination.joinpath("README.md").write_text(
        "# Engineering validation only\n\nThese archived outputs used synthetic data before the user selected A shares.\n"
        "They are not evidence about real A-share signals or deployable profitability.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
