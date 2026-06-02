"""Tests for the generic Registry[T] class."""

import pytest

from jaxgam.registry import Registry


class Dummy:
    """Trivial class for testing registry behavior."""


class DummyA(Dummy):
    pass


class DummyB(Dummy):
    pass


@pytest.fixture
def registry() -> Registry[Dummy]:
    return Registry({"alpha": DummyA, "beta": DummyB}, name="dummy")


@pytest.fixture
def cached_registry() -> Registry[Dummy]:
    return Registry(
        {"alpha": DummyA, "beta": DummyB},
        name="dummy",
        cache_instances=True,
    )


class TestGetClass:
    def test_valid_key(self, registry: Registry[Dummy]) -> None:
        assert registry.get_class("alpha") is DummyA
        assert registry.get_class("beta") is DummyB

    def test_unknown_key_raises_key_error(self, registry: Registry[Dummy]) -> None:
        with pytest.raises(KeyError, match="Unknown dummy"):
            registry.get_class("gamma")

    def test_case_insensitive(self, registry: Registry[Dummy]) -> None:
        assert registry.get_class("ALPHA") is DummyA
        assert registry.get_class("Alpha") is DummyA
        assert registry.get_class("aLpHa") is DummyA


class TestGetInstance:
    def test_returns_instance(self, registry: Registry[Dummy]) -> None:
        instance = registry.get_instance("alpha")
        assert isinstance(instance, DummyA)

    def test_uncached_returns_different_objects(
        self, registry: Registry[Dummy]
    ) -> None:
        a = registry.get_instance("alpha")
        b = registry.get_instance("alpha")
        assert a is not b

    def test_cached_returns_same_object(self, cached_registry: Registry[Dummy]) -> None:
        a = cached_registry.get_instance("alpha")
        b = cached_registry.get_instance("alpha")
        assert a is b

    def test_cached_different_keys_different_objects(
        self, cached_registry: Registry[Dummy]
    ) -> None:
        a = cached_registry.get_instance("alpha")
        b = cached_registry.get_instance("beta")
        assert a is not b

    def test_unknown_key_raises_key_error(self, registry: Registry[Dummy]) -> None:
        with pytest.raises(KeyError, match="Unknown dummy"):
            registry.get_instance("gamma")


class TestIntrospection:
    def test_available(self, registry: Registry[Dummy]) -> None:
        assert registry.available == ("alpha", "beta")

    def test_len(self, registry: Registry[Dummy]) -> None:
        assert len(registry) == 2


class DummyC(Dummy):
    pass


class TestRegister:
    def test_register_new_key(self, registry: Registry[Dummy]) -> None:
        registry.register("gamma", DummyC)
        assert registry.get_class("gamma") is DummyC

    def test_register_case_insensitive(self, registry: Registry[Dummy]) -> None:
        registry.register("Gamma", DummyC)
        assert registry.get_class("gamma") is DummyC
        assert "GAMMA" in registry

    def test_register_duplicate_builtin_raises(self, registry: Registry[Dummy]) -> None:
        with pytest.raises(ValueError, match="already registered"):
            registry.register("alpha", DummyC)

    def test_register_duplicate_custom_raises(self, registry: Registry[Dummy]) -> None:
        registry.register("gamma", DummyC)
        with pytest.raises(ValueError, match="already registered"):
            registry.register("gamma", DummyC)

    def test_register_updates_available(self, registry: Registry[Dummy]) -> None:
        registry.register("gamma", DummyC)
        assert "gamma" in registry.available

    def test_register_updates_len(self, registry: Registry[Dummy]) -> None:
        original_len = len(registry)
        registry.register("gamma", DummyC)
        assert len(registry) == original_len + 1

    def test_register_get_instance(self, registry: Registry[Dummy]) -> None:
        registry.register("gamma", DummyC)
        instance = registry.get_instance("gamma")
        assert isinstance(instance, DummyC)

    def test_register_cached_instance(self, cached_registry: Registry[Dummy]) -> None:
        cached_registry.register("gamma", DummyC)
        a = cached_registry.get_instance("gamma")
        b = cached_registry.get_instance("gamma")
        assert a is b

    def test_builtins_unchanged_after_register(self, registry: Registry[Dummy]) -> None:
        registry.register("gamma", DummyC)
        assert registry.get_class("alpha") is DummyA
        assert registry.get_class("beta") is DummyB


class TestSmoothRegistryExports:
    """Guard for U10: GP/RE smooths importable from jaxgam.smooths + registered.

    The hardening commits added GaussianProcessSmooth / RandomEffectSmooth to
    smooths/__init__.__all__; this pins the public-export contract so a future
    change cannot silently drop them (the original U10 failure mode).
    """

    EXPECTED_KEYS = frozenset({"tp", "ts", "cr", "cs", "cc", "gp", "re", "te", "ti"})

    def test_gp_re_importable_and_registered(self) -> None:
        from jaxgam.smooths import (
            GaussianProcessSmooth,
            RandomEffectSmooth,
            smooth_registry,
        )
        from jaxgam.smooths import __all__ as smooths_all
        from jaxgam.smooths.gaussian_process import (
            GaussianProcessSmooth as GPCanonical,
        )
        from jaxgam.smooths.random_effects import (
            RandomEffectSmooth as RECanonical,
        )

        assert "GaussianProcessSmooth" in smooths_all
        assert "RandomEffectSmooth" in smooths_all
        assert GaussianProcessSmooth is GPCanonical
        assert RandomEffectSmooth is RECanonical
        assert set(smooth_registry.available) == set(self.EXPECTED_KEYS)
        assert len(smooth_registry) == 9
        assert smooth_registry.get_class("gp") is GPCanonical
        assert smooth_registry.get_class("re") is RECanonical
