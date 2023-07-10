# Third Party
from ray.cluster_utils import Cluster
import pytest
import ray


def create_ray_cluster() -> Cluster:
    # First kill any other ray cluster
    if ray.is_initialized():
        ray.shutdown()

    return Cluster(
        initialize_head=True,
        head_node_args={"num_cpus": 1},
    )


@pytest.fixture(scope="session")
def mock_ray_cluster():
    cluster = create_ray_cluster()
    yield cluster

    # This code gets executed at tear down
    ray.shutdown()
