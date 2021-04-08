"""This is the main file that defines TFX pipeline and various components in the pipeline."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import datetime
from absl import logging
from typing import Any, Dict, List, Optional, Text

import tensorflow_model_analysis as tfma
from tfx.components import Evaluator
from tfx.components import ExampleValidator  # pylint: disable=unused-import
from tfx.components import Pusher
from tfx.components import ResolverNode
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.components.trainer import executor as trainer_executor
from tfx.dsl.components.base import executor_spec
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.extensions.google_cloud_ai_platform.pusher import (
    executor as ai_platform_pusher_executor,
)
from tfx.extensions.google_cloud_ai_platform.trainer import (
    executor as ai_platform_trainer_executor,
)
from tfx.extensions.google_cloud_big_query.example_gen.component import (
    BigQueryExampleGen,
)  # pylint: disable=unused-import
from tfx.orchestration import pipeline
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.proto import example_gen_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing
from tfx.utils.dsl_utils import external_input

from ml_metadata.proto import metadata_store_pb2

from utils.query_utils import load_query_string


def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    query: Text,
    preprocessing_fn: Text,
    run_fn: Text,
    train_args: trainer_pb2.TrainArgs,
    eval_args: trainer_pb2.EvalArgs,
    model_serve_dir: Text,
    metadata_connection_config: Optional[metadata_store_pb2.ConnectionConfig] = None,
    beam_pipeline_args: Optional[Dict[Text, Any]] = None,
    ai_platform_training_args: Optional[Dict[Text, Text]] = None,
    ai_platform_serving_args: Optional[Dict[Text, Any]] = None,
    enable_cache: Optional[bool] = False,
    system_config: Optional[Dict[Text, Any]] = None,
    model_config: Optional[Dict[Text, Any]] = None,
) -> pipeline.Pipeline:
    """Implements the pipeline with TFX."""

    components = []
    # %%
    # ExampleGen: Load the graph data from bigquery
    query_str = load_query_string(
        query,
        field_dict={"GOOGLE_CLOUD_PROJECT": system_config["GOOGLE_CLOUD_PROJECT"], "DEBUG_SETTINGS": model_config["query_debug_settings"]},
    )

    output_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[  # Generate no splitting, as we need to load everything
                example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=1),
            ],
        )
    )

    example_gen = BigQueryExampleGen(query=query_str, output_config=output_config)
    components.append(example_gen)

    # %%
    # StatisticsGen:
    # Computes statistics over data for visualization and example validation.
    statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])
    components.append(statistics_gen)

    # %%
    # SchemaGen:
    # Generates schema based on statistics files.
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"], infer_feature_shape=True
    )
    components.append(schema_gen)

    # %%
    # ExampleValidator:
    # Performs anomaly detection based on statistics and data schema.
    # example_validator = ExampleValidator(  # pylint: disable=unused-variable
    #     statistics=statistics_gen.outputs["statistics"],
    #     schema=schema_gen.outputs["schema"],
    # )
    # components.append(example_validator)

    # %%
    # Transform:
    # Performs transformations and feature engineering in training and serving.
    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        preprocessing_fn=preprocessing_fn,
    )
    components.append(transform)

    # %%
    # Trainer
    # Uses user-provided Python function that implements a model using TF-Learn.
    trainer_args = {
        "run_fn": run_fn,
        "transformed_examples": transform.outputs["transformed_examples"],
        "schema": schema_gen.outputs["schema"],
        "transform_graph": transform.outputs["transform_graph"],
        "train_args": train_args,
        "eval_args": eval_args,
        "custom_config": {"model_config": model_config, "system_config": system_config},
        "custom_executor_spec": executor_spec.ExecutorClassSpec(
            trainer_executor.GenericExecutor
        ),
    }

    if ai_platform_training_args is not None:
        trainer_args["custom_executor_spec"] = executor_spec.ExecutorClassSpec(
            ai_platform_trainer_executor.GenericExecutor
        )

        trainer_args["custom_config"][
            ai_platform_trainer_executor.TRAINING_ARGS_KEY
        ] = ai_platform_training_args

        # Lowercase and replace illegal characters in labels.
        # This is the job ID that will be shown in the platform
        # See https://cloud.google.com/compute/docs/naming-resources.
        trainer_args["custom_config"][ai_platform_trainer_executor.JOB_ID_KEY] = (
           "tfx_{}_{}".format(
               re.sub(r"[^a-z0-9\_]", "_", pipeline_name.lower()),
               datetime.datetime.now().strftime("%Y%m%d%H%M%S")
           )[-63:]
        )
    logging.info("trainer arguments")
    logging.info(trainer_args)

    trainer = Trainer(**trainer_args)
    components.append(trainer)

    # %%
    # ResolveNode:
    # Get the latest blessed model for model validation.
    model_resolver = ResolverNode(
        instance_name="latest_blessed_model_resolver",
        resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing),
    )
    # components.append(model_resolver)

    # %%
    # Evaluator:
    # Uses TFMA to compute a evaluation statistics over features of a model and
    # perform quality validation of a candidate model (compared to a baseline).
    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key="big_tipper")],
        slicing_specs=[tfma.SlicingSpec()],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(
                        class_name="BinaryAccuracy",
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={"value": 0.1}
                            ),
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                absolute={"value": -1e-10},
                            ),
                        ),
                    )
                ]
            )
        ],
    )
    evaluator = Evaluator(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
        # Change threshold will be ignored if there is no baseline (first run).
        eval_config=eval_config,
    )
    # components.append(evaluator)

    # %%
    # Pusher:
    # Checks whether the model passed the validation steps and pushes the model
    # to a file destination if check passed.
    pusher_args = {
        "model": trainer.outputs["model"],
        "model_blessing": evaluator.outputs["blessing"],
        "push_destination": pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=model_serve_dir,
            )
        ),
    }
    if ai_platform_serving_args is not None:
        pusher_args.update(
            {
                "custom_executor_spec": executor_spec.ExecutorClassSpec(
                    ai_platform_pusher_executor.Executor
                ),
                "custom_config": {
                    ai_platform_pusher_executor.SERVING_ARGS_KEY: ai_platform_serving_args
                },
            }
        )
    pusher = Pusher(**pusher_args)  # pylint: disable=unused-variable
    # components.append(pusher)
    
    # %%
    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=enable_cache,
        metadata_connection_config=metadata_connection_config,
        beam_pipeline_args=([f"--{key}={val}" for key, val in beam_pipeline_args.items()] if beam_pipeline_args is not None else None)
    )
