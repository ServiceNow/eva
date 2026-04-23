"""Registry for managing available metrics."""

from typing import Any

from eva.metrics.base import BaseMetric
from eva.utils.logging import get_logger

logger = get_logger(__name__)


class MetricRegistry:
    """Registry for metrics that can be used in evaluation.

    Metrics are registered by name and can be instantiated with configuration.
    """

    def __init__(self):
        self._metrics: dict[str, type[BaseMetric]] = {}

    def register(self, metric_class: type[BaseMetric]) -> type[BaseMetric]:
        """Register a metric class.

        Can be used as a decorator:

            @registry.register
            class MyMetric(BaseMetric):
                ...

        Args:
            metric_class: The metric class to register

        Returns:
            The registered class (for decorator use)
        """
        name = metric_class.name
        if name in self._metrics:
            logger.warning(f"Metric '{name}' already registered, overwriting")
        self._metrics[name] = metric_class
        logger.debug(f"Registered metric: {name}")
        return metric_class

    def get(self, name: str) -> type[BaseMetric] | None:
        """Get a metric class by name.

        Args:
            name: Name of the metric

        Returns:
            The metric class or None if not found
        """
        return self._metrics.get(name)

    def create(
        self,
        name: str,
        config: dict[str, Any] | None = None,
    ) -> BaseMetric | None:
        """Create a metric instance by name.

        Args:
            name: Name of the metric
            config: Optional configuration for the metric

        Returns:
            Metric instance or None if not found
        """
        metric_class = self.get(name)
        if metric_class is None:
            logger.warning(f"Metric '{name}' not found")
            return None
        return metric_class(config=config)

    def list_metrics(self) -> list[str]:
        """Get list of all registered metric names."""
        return list(self._metrics.keys())

    def get_all(self) -> dict[str, type[BaseMetric]]:
        """Get all registered metrics."""
        return self._metrics.copy()


# Global registry instance
_global_registry = MetricRegistry()


def get_global_registry() -> MetricRegistry:
    """Get the global metric registry."""
    return _global_registry


def register_metric(metric_class: type[BaseMetric]) -> type[BaseMetric]:
    """Register a metric in the global registry.

    Decorator for metric classes.
    """
    return _global_registry.register(metric_class)
