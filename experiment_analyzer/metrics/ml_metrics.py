from experiment_analyzer.metrics.base import Metric


class ValidationAccuracyMetric(Metric):
    @property
    def name(self) -> str:
        return "validation_accuracy"