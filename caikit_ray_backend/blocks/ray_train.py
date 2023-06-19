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

# Standard
import base64
import pickle
import uuid
from typing import Optional

# Third Party
from ray.job_submission import JobStatus

# First Party
import alog
from caikit.core.toolkit.errors import error_handler
from caikit.core.modules import ModuleBase
from caikit_ray_backend.base import SharedTrainBackendBase
from caikit_ray_backend.ray_backend import RayBackend


logger = alog.use_channel("<RYT_BLK>")
error = error_handler.get(logger)


class RayJobTrainModule(RayBackend, SharedTrainBackendBase):

    def __init__(self, config: Optional[dict] = None) -> None:
        """Function to initialize the RayJobTrainModule.

        Args:
            config: dict
                This is the Ray backend configuration.
                {
                    "connection" : {"address" : "127.0.0.1:64291"},
                    "use_jobs" : true | false
                }
        """
        super().__init__(config=config)

    def __del__(self):
        return super().__del__()

    class RayTrainModelFuture(SharedTrainBackendBase.ModelFuture):

        status_map = {JobStatus.FAILED: SharedTrainBackendBase.TrainingStatus.ERRORED,
                JobStatus.PENDING: SharedTrainBackendBase.TrainingStatus.QUEUED,
                JobStatus.RUNNING: SharedTrainBackendBase.TrainingStatus.RUNNING,
                JobStatus.STOPPED: SharedTrainBackendBase.TrainingStatus.QUEUED,
                JobStatus.SUCCEEDED: SharedTrainBackendBase.TrainingStatus.COMPLETED}

        def __init__(self, ray_be: RayBackend, ray_job_id: str, save_path = None):
            self._ray_be = ray_be
            self._ray_job_id = ray_job_id
            self._id = "caikit_ray_train_" + str(uuid.uuid4())
            self._save_path = save_path

        def status_mapper(self, ray_job_status: JobStatus) -> SharedTrainBackendBase.TrainingStatus:
            return self.status_map.get(ray_job_status)

        @property
        def id(self) -> str:
            """Every model future must have a unique ID that can be used to look
            up the in-flight training
            """
            return self._id

        @property
        def save_path(self) -> Optional[str]:
            """If created with a save path, the future must expose it"""
            return self._save_path

        def get_status(self) -> SharedTrainBackendBase.TrainingStatus:
            """Every model future must be able to poll the status of the
            training job
            """

            client = self._ray_be.get_client()
            job_status = client.get_job_status(self._ray_job_id)
            return self.status_mapper(job_status)

        def cancel(self):
            """Terminate the given training"""
            client = self._ray_be.get_client()
            client.stop_job(self._ray_job_id)

        def wait(self):
            """Block until the job reaches a terminal state"""

        def load(self) -> ModuleBase:
            """A model future must be loadable with no additional arguments"""

    @staticmethod
    def _obj_to_txt(obj):
        message_bytes = pickle.dumps(obj)
        base64_bytes = base64.b64encode(message_bytes)
        txt = base64_bytes.decode('ascii')
        return txt

    def train(
        self,
        module_class: str,
        save_path: Optional[str] = None,
        num_gpus: int = None,
        num_cpus: int = None,
        *args,
        **kwargs,
    ) -> RayTrainModelFuture:
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

        ray_job_client = self.get_client()

        error.type_check("<RYT94736704E>", str, module_class=module_class)

        env_vars = {"module_class" : self._obj_to_txt(module_class)}

        if save_path:
            error.type_check("<RYT69631537E>", str, save_path=save_path)
            error.dir_check("<RYT48093065E>", save_path)
            env_vars["save_path"] = save_path
        else:
            save_path = None

        # Validate number of CPUs and GPUs requested.
        # TODO: Should we have configurable limits on number of each that can be requested?
        if num_cpus:
            error.type_check("<RYT19121015E>", int, num_cpus=num_cpus)
            error.value_check("<RYT81418146E>", "> 0", num_cpus=num_cpus)
            env_vars["num_cpus"] = num_cpus
        if num_gpus:
            error.type_check("<RYT94930817E>", int, num_gpus=num_gpus)
            error.type_check("<RYT87231812E>", "> 0", num_gpus=num_gpus)
            env_vars["num_gpus"] = num_gpus

        # Serialize **kwargs and add them to environment variables
        my_kwargs = {}
        for key, value in kwargs.items():
            my_kwargs[key] =self._obj_to_txt(value)
        env_vars["kwargs"] = my_kwargs

        # Serialize *args and add them to environment variables
        my_args = []
        for arg in args:
            my_args.append(self._obj_to_txt(arg))
        env_vars["args"] = my_args

        job_id = ray_job_client.submit_job(entrypoint="ray_submitter", runtime_env=env_vars)

        model_future = self.RayTrainModelFuture(self, job_id, save_path)

        return model_future

    def get_future(self, training_id: str) -> Optional[RayTrainModelFuture]:
        """All shared train backends must be able to retrieve the future for a
        given training by id
        """
