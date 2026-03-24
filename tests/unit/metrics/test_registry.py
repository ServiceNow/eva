"""Tests for MetricRegistry."""

from eva.metrics.base import BaseMetric, MetricContext
from eva.metrics.registry import MetricRegistry, get_global_registry, register_metric
from eva.models.results import MetricScore


class FakeMetric(BaseMetric):
    """Minimal concrete metric for testing."""

    name = "fake_metric"
    description = "A fake metric"
    metric_type = "code"

    async def compute(self, context: MetricContext) -> MetricScore:
        return MetricScore(name=self.name, score=1.0, normalized_score=1.0)


class AnotherFakeMetric(BaseMetric):
    name = "another_fake"
    description = "Another fake"
    metric_type = "code"

    async def compute(self, context: MetricContext) -> MetricScore:
        return MetricScore(name=self.name, score=0.5, normalized_score=0.5)


class TestMetricRegistry:
    def setup_method(self):
        self.registry = MetricRegistry()

    def test_register_and_get(self):
        self.registry.register(FakeMetric)
        assert self.registry.get("fake_metric") is FakeMetric

    def test_get_unknown_returns_none(self):
        assert self.registry.get("nonexistent") is None

    def test_register_returns_class_for_decorator_use(self):
        result = self.registry.register(FakeMetric)
        assert result is FakeMetric

    def test_register_overwrites_existing(self):
        self.registry.register(FakeMetric)

        class FakeMetricV2(BaseMetric):
            name = "fake_metric"
            description = "v2"
            metric_type = "code"

            async def compute(self, context):
                pass

        self.registry.register(FakeMetricV2)
        assert self.registry.get("fake_metric") is FakeMetricV2

    def test_create_returns_instance(self):
        self.registry.register(FakeMetric)
        instance = self.registry.create("fake_metric")
        assert isinstance(instance, FakeMetric)

    def test_create_with_config(self):
        self.registry.register(FakeMetric)
        instance = self.registry.create("fake_metric", config={"judge_model": "gpt-4o"})
        assert instance is not None
        assert instance.config.get("judge_model") == "gpt-4o"

    def test_create_unknown_returns_none(self):
        assert self.registry.create("nonexistent") is None

    def test_list_metrics(self):
        self.registry.register(FakeMetric)
        self.registry.register(AnotherFakeMetric)
        names = self.registry.list_metrics()
        assert set(names) == {"fake_metric", "another_fake"}

    def test_get_all_returns_copy(self):
        self.registry.register(FakeMetric)
        all_metrics = self.registry.get_all()
        all_metrics["injected"] = None  # Modify the copy
        assert "injected" not in self.registry.get_all()

    def test_empty_registry(self):
        assert self.registry.list_metrics() == []
        assert self.registry.get_all() == {}


class TestGlobalRegistry:
    def test_get_global_registry_returns_singleton(self):
        r1 = get_global_registry()
        r2 = get_global_registry()
        assert r1 is r2

    def test_register_metric_decorator(self):
        registry = get_global_registry()
        # FakeMetric may already be registered from other tests; just verify it works
        register_metric(FakeMetric)
        assert registry.get("fake_metric") is FakeMetric
