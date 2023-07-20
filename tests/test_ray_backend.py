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
# Third party
from ray import job_submission
from sample_lib.modules.sample_task.sample_implementation import SampleModule
import pytest

# First Party
from caikit.core.data_model import DataStream
from caikit.core.model_management.model_trainer_base import TrainingStatus
import caikit

# Local
from caikit_ray_backend.blocks.ray_train import RayJobTrainModule
from caikit_ray_backend.ray_backend import RayBackend
from tests.fixtures.mock_ray_cluster import mock_ray_cluster
from tests.fixtures.sample_lib.modules.sample_task.sample_implementation import (
    SampleModule,
)


@pytest.fixture
def jsonl_file_data_stream():
    fixtures_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "fixtures")
    samples_path = os.path.join(fixtures_dir, "data_stream_inputs")
    training_data = DataStream.from_jsonl(os.path.join(samples_path, "sample.jsonl"))
    return training_data


def test_job_submission_client(mock_ray_cluster, jsonl_file_data_stream):
    config = {"connection": {"address": mock_ray_cluster.address}}
    trainer = RayJobTrainModule(config, "ray_backend")

    args = [jsonl_file_data_stream]
    model_future = trainer.train(
        SampleModule,
        *args,
        save_path="/tmp",
    )
    assert model_future.id != None

    status = model_future.get_info().status
    assert status != None

    # We're going to wait until either success or failure. Given it's a sample lib, it should be fast
    # But time it out at 60 seconds just incase
    count = 0
    while True:
        time.sleep(1)
        info = model_future.get_info()
        status = info.status
        if status == TrainingStatus.ERRORED or count >= 60:
            print(info.errors)
            assert 0

        if status == TrainingStatus.COMPLETED:
            break

        count += 1

    assert info.submission_time
    assert info.completion_time
    assert info.submission_time < info.completion_time


def test_wait(mock_ray_cluster, jsonl_file_data_stream):
    config = {"connection": {"address": mock_ray_cluster.address}}
    trainer = RayJobTrainModule(config, "ray_backend")

    args = [jsonl_file_data_stream]
    model_future = trainer.train(
        SampleModule,
        *args,
        save_path="/tmp",
    )

    model_future.POLLING_INTERVAL = 1
    model_future.POLLING_TIMEOUT = 45
    model_future.wait()
    status = model_future.get_info().status
    print(status)
    assert status == TrainingStatus.COMPLETED


def test_load(mock_ray_cluster, jsonl_file_data_stream):
    config = {"connection": {"address": mock_ray_cluster.address}}
    trainer = RayJobTrainModule(config, "ray_backend")

    args = [jsonl_file_data_stream]
    model_future = trainer.train(
        SampleModule,
        *args,
        save_path="/tmp",
    )

    model_future.POLLING_INTERVAL = 1
    model_future.POLLING_TIMEOUT = 45
    model = model_future.load()
    assert isinstance(model, SampleModule)


def test_cancel(mock_ray_cluster, jsonl_file_data_stream):
    config = {"connection": {"address": mock_ray_cluster.address}}
    trainer = RayJobTrainModule(config, "ray_backend")

    args = [jsonl_file_data_stream]
    model_future = trainer.train(
        SampleModule,
        *args,
        save_path="/tmp",
    )

    model_future.cancel()
    elapsed_time = 0
    while elapsed_time < 30:
        time.sleep(1)
        status = model_future.get_info().status
        if status == TrainingStatus.CANCELED:
            assert 1
        elapsed_time += 1

    status = model_future.get_info().status
    print("Final status was", status)
    assert status == TrainingStatus.CANCELED


## Test Ray Backend


def test_ray_backend_is_registered():
    """Make sure that the Ray backend is correctly registered with caikit"""
    assert (
        RayJobTrainModule.backend_type
        in caikit.core.module_backends.backend_types.module_backend_types()
    )


def test_ray_local_backend_config_valid():
    """Make sure that the Ray backend can be configured with a valid config
    blob for an insecure server
    """
    ray_be = RayBackend({})
    ray_be.start()
    assert ray_be.is_started


def test_ray_local_backend_get_client():
    """Make sure the get_client() call implicitly starts it"""
    ray_be = RayBackend({})
    client = ray_be.get_client()
    assert ray_be.is_started


def test_get_client_after_manually_starting():
    ray_be = RayBackend({})
    ray_be.start()
    client = ray_be.get_client()
    assert ray_be.is_started


def test_get_job_submission_client(mock_ray_cluster):
    config = {"connection": {"address": mock_ray_cluster.address}}
    ray_be = RayBackend(config)
    client = ray_be.get_client()
    assert ray_be.is_started
    assert isinstance(client, job_submission.JobSubmissionClient)


## Failure Tests ###############################################################


def test_stop_cluster(mock_ray_cluster):
    config = {"connection": {"address": mock_ray_cluster.address}}
    ray_be = RayBackend(config)
    mock_ray_cluster.shutdown()
    with pytest.raises(ConnectionError):
        client = ray_be.get_client()


def test_invalid_connection():
    """Make sure that invalid connections cause errors"""
    # All forms of invalid hostname
    with pytest.raises(TypeError):
        RayBackend({"connection": "not a dict"})
    with pytest.raises(TypeError):
        RayBackend({"connection": {"not_address": "localhost"}})
