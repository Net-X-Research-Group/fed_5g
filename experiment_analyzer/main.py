\
import argparse
import logging
from pathlib import Path
import pandas as pd # Required for type hints and potentially direct use

# Project-specific imports
from experiment_analyzer.processor import Experiment
from experiment_analyzer.metrics.base import Metric # For type hinting
from experiment_analyzer.metrics.latency import LatencyMetric
from experiment_analyzer.metrics.ml_metrics import ValidationAccuracyMetric
# Import other custom metric classes here as they are developed
# e.g., from experiment_analyzer.metrics.radio_metrics import RsrpMetric

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Federated Learning Experiment Analyzer")
    parser.add_argument(
        "experiment_path",
        type=str,
        help="Path to the root directory of the experiment. This directory should contain subdirectories, each representing a configuration."
    )
    # TODO: Add arguments to specify which metrics to run, or load from a config file.
    args = parser.parse_args()

    experiment_path = Path(args.experiment_path)
    if not experiment_path.is_dir():
        logger.error(f"Experiment path {experiment_path} does not exist or is not a directory.")
        return

    logger.info(f"Starting analysis for experiment: {experiment_path.resolve()}")
    experiment = Experiment(path=experiment_path)

    # Define the metrics to be processed.
    # These should be instances of classes inheriting from Metric.
    # Ensure these metric classes are correctly implemented.
    # For example, LatencyMetric might need to be split into UplinkLatencyMetric and DownlinkLatencyMetric
    # if extract_from_trial needs to return distinct DataFrames that are processed separately.
    # For now, assuming LatencyMetric is adjusted to conform to the Metric interface.
    metrics_to_process: List[Metric] = [
        LatencyMetric(),
        ValidationAccuracyMetric(),
        # Add other metric instances here:
        # e.g., ThroughputMetric(), PacketLossMetric(), etc.
    ]

    if not metrics_to_process:
        logger.warning("No metrics defined for processing. Exiting.")
        return

    # This dictionary will store aggregated results for each metric across all configurations
    # Structure: {metric_name: {config_identifier: aggregated_df_for_that_config}}
    all_configs_aggregated_metrics: Dict[str, Dict[str, pd.DataFrame]] = {
        metric.name: {} for metric in metrics_to_process
    }

    configurations = experiment.get_configurations()
    if not configurations:
        logger.warning(f"No configurations found in experiment: {experiment.identifier}")
        return

    for config in configurations:
        logger.info(f"Processing configuration: {config.identifier}")
        config_output_path = config.get_output_path() # Ensures output dir exists
        config_figure_path = config.get_figure_path() # Ensures figures dir exists

        # Store raw trial dataframes for this configuration before aggregation
        # Structure: {metric_name: [trial1_df, trial2_df, ...]}
        current_config_trial_dfs: Dict[str, List[pd.DataFrame]] = {
            metric.name: [] for metric in metrics_to_process
        }

        trials = config.get_trials()
        if not trials:
            logger.warning(f"No trials found in configuration: {config.identifier}")
            continue

        for trial in trials:
            logger.info(f"Processing trial: {trial.identifier} in configuration: {config.identifier}")
            # trial.process_trial calls metric.extract_from_trial for each metric
            # It returns a Dict[str, Optional[pd.DataFrame]]
            trial_metric_data = trial.process_trial(metrics_to_process)

            for metric in metrics_to_process:
                df = trial_metric_data.get(metric.name)
                if df is not None and not df.empty:
                    current_config_trial_dfs[metric.name].append(df)
                else:
                    logger.warning(
                        f"Metric '{metric.name}' extract_from_trial returned no data for trial '{trial.identifier}' in config '{config.identifier}'.")

        # Aggregate metrics across trials for the current configuration
        for metric in metrics_to_process:
            trial_dfs_for_metric = current_config_trial_dfs[metric.name]
            if trial_dfs_for_metric:
                logger.info(f"Aggregating metric '{metric.name}' across {len(trial_dfs_for_metric)} trials for configuration '{config.identifier}'.")
                try:
                    # aggregate_across_trials should save its output (e.g., CSV) to config_output_path
                    # and return the aggregated DataFrame.
                    aggregated_df = metric.aggregate_across_trials(trials=trial_dfs_for_metric, config_output_path=config_output_path)

                    if aggregated_df is not None and not aggregated_df.empty:
                        all_configs_aggregated_metrics[metric.name][config.identifier] = aggregated_df

                        logger.info(f"Visualizing metric '{metric.name}' for single configuration '{config.identifier}'.")
                        # vizualize_single_config should save its output (e.g., plot) to config_figure_path
                        metric.vizualize_single_config(df=aggregated_df, output_path_str=str(config_figure_path))
                    else:
                        logger.warning(
                            f"Aggregation for metric '{metric.name}' in config '{config.identifier}' resulted in no data.")
                except Exception as e:
                    logger.error(f"Error during aggregation or visualization for metric '{metric.name}' in config '{config.identifier}': {e}", exc_info=True)
            else:
                logger.warning(
                    f"No trial data successfully extracted for metric '{metric.name}' in configuration '{config.identifier}' to aggregate.")

    # Aggregate and visualize metrics across all configurations
    experiment_output_path = experiment.get_output_path() # Ensures output dir exists
    experiment_figure_path = experiment.get_figure_path() # Ensures figures dir exists

    for metric in metrics_to_process:
        config_dfs_for_metric = all_configs_aggregated_metrics[metric.name]
        if config_dfs_for_metric:
            logger.info(f"Aggregating metric '{metric.name}' across {len(config_dfs_for_metric)} configurations for experiment '{experiment.identifier}'.")
            try:
                # aggregate_across_configs should save its output to experiment_output_path
                # and return the cross-configuration aggregated DataFrame.
                cross_config_df = metric.aggregate_across_configs(config_dfs=config_dfs_for_metric, experiment_output_path=experiment_output_path)

                if cross_config_df is not None and not cross_config_df.empty:
                    logger.info(f"Visualizing metric '{metric.name}' across configurations for experiment '{experiment.identifier}'.")
                    # vizualize_across_configs should save its output to experiment_figure_path
                    # It takes the same dict that aggregate_across_configs took, or the result of it, depending on implementation.
                    # The base class signature is (self, dfs: Dict[str, pd.DataFrame], output_path: str)
                    # So we pass config_dfs_for_metric
                    metric.vizualize_across_configs(dfs=config_dfs_for_metric, output_path_str=str(experiment_figure_path))
                else:
                    logger.warning(
                        f"Cross-configuration aggregation for metric '{metric.name}' in experiment '{experiment.identifier}' resulted in no data.")
            except Exception as e:
                logger.error(f"Error during cross-configuration aggregation or visualization for metric '{metric.name}': {e}", exc_info=True)
        else:
            logger.warning(
                f"No configuration data found for metric '{metric.name}' to aggregate or visualize across configurations for experiment '{experiment.identifier}'.")

    logger.info(f"Experiment analysis complete for: {experiment.identifier}")

if __name__ == "__main__":
    main()

