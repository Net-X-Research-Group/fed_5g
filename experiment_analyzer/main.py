from pathlib import Path

import logging_setup
from experiment_analyzer.metrics.latency import LatencyMetric
from experiment_analyzer.metrics.ml_metrics import MLMetric
from experiment_analyzer.metrics.physical_layer import PHYMetric
from experiment_analyzer.metrics.base import Metric
from experiment_analyzer.data_models import Trial, Configuration, Experiment
from typing import List, Dict
import pandas as pd

logger = logging_setup.setup_logging('info')

def process_trial(trial: Trial, metrics: List[Metric]) -> Dict[str, pd.DataFrame]:
    results = {}
    figure_path = trial.get_figure_path()
    for metric in metrics:
        df = metric.extract_from_trial(trial)
        metric.visualize_trial(df, figure_path)
        results[metric.name] = df
    return results


def process_configuration(config: Configuration, trial_data: Dict, metrics: List[Metric]) -> Dict:
    figure_path = config.get_figure_path()
    aggregated_metrics = {}
    for metric in metrics:
        metric_data_over_trials = []
        for trial_id, trial_metrics in trial_data.items():
            if metric.name in trial_metrics:
                metric_data_over_trials.append(trial_metrics[metric.name])

        if metric_data_over_trials:
            aggregated_df = metric.aggregate_across_trials(config, metric_data_over_trials)
            aggregated_metrics[metric.name] = aggregated_df
            metric.visualize_trial(aggregated_df, figure_path)
    return aggregated_metrics


def process_experiment(experiment: Experiment, configuration_data: Dict, metrics: List[Metric]) -> None:
    figure_path = experiment.get_figure_path()
    for metric in metrics:
        experiment_data = {}
        for configuration_id, configuration_metrics in configuration_data.items():
            if metric.name in configuration_metrics:
                experiment_data[configuration_id] = configuration_metrics[metric.name]
        output_path = experiment.get_output_path()
        experiment_aggregated = metric.aggregate_across_configs(experiment_data, output_path)
        metric.visualize_across_configs(experiment_aggregated, figure_path)


def main():
    input_dir = Path('/Users/roberthayek/5G_Test/')

    metrics = [LatencyMetric(),
               MLMetric('validation_accuracy'),
               MLMetric('train_accuracy'),
               MLMetric('train_loss'),
               MLMetric('validation_loss'),
               PHYMetric('rsrp'),
               PHYMetric('rsrq'),
               PHYMetric('rssi')
               ]

    if not input_dir.exists():
        logger.error('Input directory does not exist')
        return

    experiment = Experiment(path=input_dir)

    configurations_all_metrics = {}
    for configuration in experiment.get_configurations():
        logger.info(f'Processing configuration: {configuration.identifier}')
        trials_all_metrics = {}
        for trial in configuration.get_trials():
            logger.info(f'Processing trial: {trial.identifier}')
            trials_all_metrics[trial.identifier] = process_trial(trial, metrics)
        configurations_all_metrics[configuration.identifier] = process_configuration(configuration, trials_all_metrics, metrics)

    process_experiment(experiment, configurations_all_metrics, metrics)

    print('stop')


if __name__ == '__main__':
    main()


