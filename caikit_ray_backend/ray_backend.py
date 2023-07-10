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
"""This module implements a TGIS backend configuration
"""

# Standard
from typing import Dict, Optional

# Third Party
from ray.job_submission import JobSubmissionClient
import ray

# First Party
from caikit.core.module_backends.backend_types import register_backend_type
from caikit.core.module_backends.base import BackendBase
from caikit.core.toolkit.errors import error_handler
import alog

log = alog.use_channel("RAYBKND")
error = error_handler.get(log)

# pylint: disable=too-many-instance-attributes
class RayBackend(BackendBase):
    """Caikit backend with a connection to a Ray cluster. If no connection
    details are given, this backend will use the default Ray init, which connects
    to a local instance of Ray.
    """

    backend_type = "RAY"

    def __init__(self, config: Optional[dict] = None):
        """
        This method will launch a Ray job which will in turn call the train() and save() methods
        on the given module identified by module_class.


        Args:
            module_class: str
                Fully qualified path of module class
                i.e. "caikit_example.modules.example_module.ExampleClass
                This module should extend ModuleBase and implement train() and save() methods

            save_path: str (Optional)
                Location on disk of where to save the model

            num_gpus: int (Optional)
                The number of gpus to be used for the training task

            num_cpus: int (Optional)
                The number of cpus to be used for the training task

            *args
                Positional arguments for the train method

            **kwargs
                Named arguments for the train method

        Returns:
            RayTrainModelFuture
        """
        super().__init__(config=config)

        self._client = None
        self._local_ray = None

        # Parse the config to see if we"re managing a connection to a remote
        # Ray instance or running a local copy
        connection_cfg = self.config.get("connection")
        local_cfg = self.config.get("local") or {}

        if not connection_cfg:
            log.info("<RBE20235227I>", "Managing local Ray instance")
            error.type_check("<RBE20235227I>", dict, local=local_cfg)
            self._local_ray = True

        else:
            log.info("<RBE20235226I>", "Managing remote Ray connection")
            error.type_check(
                "<RBE20235229E>", dict, allow_none=True, connection=connection_cfg
            )
            self._local_ray = False
            self._address = connection_cfg.get("address")
            error.type_check("<TGB20235230E>", str, address=self._address)
            error.value_check(
                "<TGB20235231E>",
                ":" in self._address,
                "Invalid address: %s",
                self._address,
            )

    def __del__(self):
        self.stop()

    # pylint: disable=unused-argument
    def register_config(self, config: Dict) -> None:
        """Function to merge configs with existing configurations"""
        error(
            "<RBE20236213E>",
            AssertionError(
                f"{self.backend_type} backend does not support this operation"
            ),
        )

    def start(self):
        """Start backend, initializing the client"""
        # if self._local_ray: #and not ray.is_initialized():
        # ray.init(ignore_reinit_error=True, include_dashboard=True)
        self._setup_job_client()
        self._started = True

    def stop(self):
        """Stop backend"""
        ray.shutdown()
        self._started = False
        self._client = None

    @property
    def local_ray(self) -> bool:
        return self._local_ray

    def get_client(self):

        if not self._client:
            self.start()

        return self._client

    def _setup_job_client(self):
        if self._client:
            log.warning(
                "<RBE20236241W>",
                "Ray job client already initialized, no further action taken.",
            )
            return

        if self._local_ray:
            log.info("<RBE20236430I>", "Initializing job client to local instance")
            self._client = JobSubmissionClient(create_cluster_if_needed=True)
        else:
            log.info("<RBE20236431I>", "Initializing job client to [%s]", self._address)
            self._client = JobSubmissionClient(address=self._address)


# Register local backend
register_backend_type(RayBackend)
