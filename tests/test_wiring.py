"""Both repos are importable from their own checkouts, with no shadowing."""

from pathlib import Path

from conftest import ENGINE, EXPECTED_ENGINE, REPO_ROOT


def test_scenario_generation_comes_from_this_repo():
    import scenario_generation

    path = Path(scenario_generation.__file__).resolve()
    assert path.is_relative_to(REPO_ROOT)


def test_server_comes_from_sibling_engine_checkout():
    import server

    path = Path(server.__file__).resolve()
    assert path.is_relative_to(EXPECTED_ENGINE)


def test_backend_finds_engine_root():
    assert ENGINE == EXPECTED_ENGINE


def test_engine_modules_are_importable():
    from server import airegistry
    from server.map import MapData  # noqa: F401

    assert "pass-agg" in airegistry.ai_registry
