# Copyright The Caikit Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for Ray backend training
"""
# Standard
import os
import time

# Third Party
from ray.cluster_utils import Cluster

# Third party
import pytest
import ray

# First Party
from caikit.core.data_model import DataStream

# Local
from caikit_ray_backend.base import SharedTrainBackendBase
from caikit_ray_backend.blocks.ray_train import RayJobTrainModule
from tests.fixtures.mock_ray_cluster import mock_ray_cluster


@pytest.fixture
def jsonl_file_data_stream():
    fixtures_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "fixtures")
    samples_path = os.path.join(fixtures_dir, "data_stream_inputs")
    training_data = DataStream.from_jsonl(os.path.join(samples_path, "sample.jsonl"))
    return training_data


def test_job_submission_client(mock_ray_cluster, jsonl_file_data_stream):
    config = {"connection": {"address": mock_ray_cluster.address}}
    trainer = RayJobTrainModule(config)

    args = [jsonl_file_data_stream]
    model_future = trainer.train(
        "tests.fixtures.sample_lib.modules.sample_task.sample_implementation.SampleModule",
        *args,
        save_path="/tmp",
    )
    assert model_future.id != None

    status = model_future.get_status()
    assert status != None

    # We're going to wait until either success or failure. Given it's a sample lib, it should be fast
    # But time it out at 60 seconds just incase
    count = 0
    while True:
        time.sleep(1)
        status = model_future.get_status()
        if status == SharedTrainBackendBase.TrainingStatus.ERRORED or count >= 60:
            print(
                model_future._ray_be.get_client().get_job_logs(model_future._ray_job_id)
            )
            assert 0

        if status == SharedTrainBackendBase.TrainingStatus.COMPLETED:
            break

        count += 1


## Error Cases


def test_invalid_type_model_save_path(mock_ray_cluster, jsonl_file_data_stream):
    config = {"connection": {"address": mock_ray_cluster.address}}
    trainer = RayJobTrainModule(config)

    kwargs = {"training data": jsonl_file_data_stream}
    with pytest.raises(TypeError):
        model_future = trainer.train(
            save_path=["this is a list"],
            module_class="fixtures.sample_lib.modules.sample_task.sample_implementation.SampleModule",
            **kwargs,
        )


def test_invalid_path_model_save(mock_ray_cluster, jsonl_file_data_stream):
    # This should pass the inital file check but the ray job will fail
    config = {"connection": {"address": mock_ray_cluster.address}}
    trainer = RayJobTrainModule(config)
    kwargs = {"training data": jsonl_file_data_stream}
    with pytest.raises(FileNotFoundError):
        model_future = trainer.train(
            save_path="/bogus_path",
            module_class="fixtures.sample_lib.modules.sample_task.sample_implementation.SampleModule",
            **kwargs,
        )
