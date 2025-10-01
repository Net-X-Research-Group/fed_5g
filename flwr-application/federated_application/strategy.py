import csv
import io
import logging
import os
import time
from logging import INFO
from pathlib import Path
from typing import Optional, Callable

import wandb
from flwr.common import ArrayRecord, MetricRecord, ConfigRecord
from flwr.common import log
from flwr.server import Grid
from flwr.serverapp.strategy import FedAvg, Result
from flwr.serverapp.strategy.strategy_utils import log_strategy_start_info, aggregate_arrayrecords

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_NAME = "Pytorch-5G-FLWR-CIFAR10"


class CellFedAvg(FedAvg):
    def configure_train(self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid):
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(self, server_round, replies):
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        arrays, metrics = None, None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]

            # Aggregate ArrayRecords
            arrays = aggregate_arrayrecords(
                reply_contents,
                self.weighted_by_key,
            )

            # Aggregate MetricRecords
            metrics = self.train_metrics_aggr_fn(
                reply_contents,
                self.weighted_by_key,
                server_round,
                'train'
            )
        return arrays, metrics

    def configure_evaluate(self, server_round, arrays, config, grid):
        return super().configure_evaluate(server_round, arrays, config, grid)

    def aggregate_evaluate(self, server_round, replies):
        """Aggregate MetricRecords in the received Messages."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=False)

        metrics = None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]

            # Aggregate MetricRecords
            metrics = self.evaluate_metrics_aggr_fn(
                reply_contents,
                self.weighted_by_key,
                server_round,
                'evaluate'
            )
        return metrics

    def summary(self):
        super().summary()

    def set_save_path(self, path: Path):
        self.save_path = path

    def write_agg_metric(self, results, server_round, metric_type):
        """
        Helper Function
        -----------------
        Writes aggregated and individual metrics to a JSON file.
        """
        output_path = self.save_path / f'{metric_type}_agg_metrics.csv'
        file_exists = os.path.isfile(output_path)
        headers = ['server_round'] + list(results.keys())
        row = [server_round] + list(results.values())

        with open(output_path, 'a+', newline='') as f:  # a+ is open file for updating
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            writer.writerow(row)

    def start(
            self,
            grid: Grid,
            initial_arrays: ArrayRecord,
            num_rounds: int = 3,
            timeout: float = 3600,
            enable_wandb: bool = False,
            train_config: Optional[ConfigRecord] = None,
            evaluate_config: Optional[ConfigRecord] = None,
            evaluate_fn: Optional[
                Callable[[int, ArrayRecord], Optional[MetricRecord]]
            ] = None,
    ) -> Result:
        # Init wandb
        if enable_wandb:
            name = f"{str(self.save_path.parent.name)}/{str(self.save_path.name)}-ServerApp"
            wandb.init(project=PROJECT_NAME, name=name)
        log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log_strategy_start_info(
            num_rounds, initial_arrays, train_config, evaluate_config
        )
        self.summary()
        log(INFO, "")

        # Initialize if None
        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = Result()

        t_start = time.time()
        # Evaluate starting global parameters
        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            log(INFO, "Initial global evaluation results: %s", res)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res

        arrays = initial_arrays

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)

            # -----------------------------------------------------------------
            # --- TRAINING (CLIENTAPP-SIDE) -----------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure training round
            # Send messages and wait for replies
            train_replies = grid.send_and_receive(
                messages=self.configure_train(
                    current_round,
                    arrays,
                    train_config,
                    grid,
                ),
                timeout=timeout,
            )

            # Aggregate train
            agg_arrays, agg_train_metrics = self.aggregate_train(
                current_round,
                train_replies,
            )

            # Log training metrics and append to history
            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays
            if agg_train_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics
                if enable_wandb:
                    wandb.log(dict(agg_train_metrics), step=current_round)
                self.write_agg_metric(agg_train_metrics, current_round, 'train')

            # -----------------------------------------------------------------
            # --- EVALUATION (CLIENTAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure evaluation round
            # Send messages and wait for replies
            evaluate_replies = grid.send_and_receive(
                messages=self.configure_evaluate(
                    current_round,
                    arrays,
                    evaluate_config,
                    grid,
                ),
                timeout=timeout,
            )

            # Aggregate evaluate
            agg_evaluate_metrics = self.aggregate_evaluate(
                current_round,
                evaluate_replies,
            )

            # Log training metrics and append to history
            if agg_evaluate_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_evaluate_metrics)
                result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics
                if enable_wandb:
                    wandb.log(dict(agg_evaluate_metrics), step=current_round)
                self.write_agg_metric(agg_evaluate_metrics, current_round, 'evaluate')

        # Save the wall-clock time
        execution_time = time.time() - t_start
        execution_time_path = self.save_path / 'execution_time.txt'
        with open(execution_time_path, 'w') as f:
            f.write(f"{execution_time:.2f}s\n")

        log(INFO, "")
        log(INFO, "Strategy execution finished in %.2fs", execution_time)
        log(INFO, "")
        log(INFO, "Final results:")
        log(INFO, "")
        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))
        log(INFO, "")

        return result
