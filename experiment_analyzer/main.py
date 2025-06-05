from pathlib import Path
import logging
import colorlog

from experiment_analyzer.metrics.latency import LatencyMetric
from experiment_analyzer.metrics.ml_metrics import MLMetric
from experiment_analyzer.metrics.physical_layer import PHYMetric
from experiment_analyzer.processor import Experiment

handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
))

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [handler]


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
            output_path = trial.get_output_path()
            trials_all_metrics[trial.identifier] = trial.process_trial(output_path, metrics)
        configurations_all_metrics[configuration.identifier] = configuration.aggregate_trials(trials_all_metrics, metrics)

    experiment.process(configurations_all_metrics, metrics)

    print('stop')


if __name__ == '__main__':
    main()


