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
Unit tests for Ray backend
"""

# Third Party
import pytest

# First Party
import caikit
from ray import job_submission

# Local
from caikit_ray_backend.ray_backend import RayBackend
from tests.fixtures.mock_ray_cluster import mock_ray_cluster


## Happy Path Tests ############################################################


def test_ray_backend_is_registered():
    """Make sure that the Ray backend is correctly registered with caikit"""
    assert (
        RayBackend.backend_type
        in caikit.core.module_backends.backend_types.module_backend_types()
    )

def test_get_job_submission_client(mock_ray_cluster):
    config = {"connection": {"address": mock_ray_cluster.address}}
    ray_be = RayBackend(config=config)
    client = ray_be.get_client()
    assert ray_be.is_started
    assert isinstance(client, job_submission.JobSubmissionClient)


## Failure Tests ###############################################################


def test_stop_cluster(mock_ray_cluster):
    config = {"connection": {"address": mock_ray_cluster.address}}
    ray_be = RayBackend(config=config)
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
