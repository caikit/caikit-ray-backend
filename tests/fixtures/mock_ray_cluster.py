# Third Party
from ray.cluster_utils import Cluster
import pytest
import ray


def create_ray_cluster() -> Cluster:
    # First kill any other ray cluster
    ray.shutdown()

    return Cluster(
        initialize_head=True,
        head_node_args={"num_cpus": 1, "include_dashboard": True},
    )


@pytest.fixture(scope="session")
def mock_ray_cluster():
    print("**Creating cluster")
    cluster = create_ray_cluster()
    ray.init(address=cluster.address)
    return cluster
