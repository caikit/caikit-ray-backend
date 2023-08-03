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
from typing import Optional, Type
import base64
import pickle
import time

# Third Party
from ray.job_submission import JobStatus

# First Party
from caikit.core.model_management import ModelTrainerBase, model_trainer_factory
from caikit.core.model_management.model_trainer_base import TrainingInfo, TrainingStatus
from caikit.core.modules import ModuleBase
from caikit.core.toolkit.errors import error_handler
import aconfig
import alog
import caikit

# Local
from caikit_ray_backend.ray_backend import RayBackend

logger = alog.use_channel("<RYT_BLK>")
error = error_handler.get(logger)


class RayJobTrainModule(ModelTrainerBase, RayBackend):
    name = "RAY_JOB_TRAIN"

    def __init__(self, config: aconfig.Config, instance_name: str):
        """Function to initialize the RayJobTrainModule.

        Args:
            config: dict
                This is the Ray backend configuration.
                {
                    "connection" : {"address" : "127.0.0.1:64291"},
                }
        """
        self._instance_name = instance_name
        self.config = config
        super().__init__(config=config, instance_name=instance_name)
        self._init_connections()

    class RayTrainModelFuture(ModelTrainerBase.ModelFutureBase):

        POLLING_INTERVAL = 30
        POLLING_TIMEOUT = 7200

        status_map = {
            JobStatus.FAILED: TrainingStatus.ERRORED,
            JobStatus.PENDING: TrainingStatus.QUEUED,
            JobStatus.RUNNING: TrainingStatus.RUNNING,
            JobStatus.STOPPED: TrainingStatus.CANCELED,
            JobStatus.SUCCEEDED: TrainingStatus.COMPLETED,
        }

        def __init__(
            self,
            ray_be: RayBackend,
            ray_job_id: str,
            save_with_id: bool,
            save_path: Optional[str],
        ):
            self._ray_be = ray_be
            self._ray_job_id = ray_job_id
            super().__init__(
                trainer_name=self.__getattribute__.__name__,
                training_id=ray_job_id,
                save_with_id=save_with_id,
                save_path=save_path,
            )

        def status_mapper(self, ray_job_status: JobStatus) -> TrainingStatus:
            return self.status_map.get(ray_job_status)

        def get_info(self) -> TrainingInfo:
            """Every model future must be able to poll the status of the
            training job
            """

            client = self._ray_be.get_client()
            ray_job_info = client.get_job_info(self._ray_job_id)
            job_status = self.status_mapper(ray_job_info.status)
            error_info = None

            if ray_job_info.status in [JobStatus.FAILED, JobStatus.STOPPED]:
                error_info = [ray_job_info.error_type, ray_job_info.message]

            return TrainingInfo(job_status, error_info)

        def cancel(self):
            """Terminate the given training"""
            client = self._ray_be.get_client()
            client.stop_job(self._ray_job_id)

        def wait(self):
            """Block until the job reaches a terminal state"""
            logger.info("Waiting for job [%s] to complete", self.id)
            elapsed_time = 0
            while elapsed_time < self.POLLING_TIMEOUT:
                logger.debug("Polling for job [%s] status", self.id)
                self._validate_success()
                time.sleep(
                    min(self.POLLING_INTERVAL, self.POLLING_TIMEOUT - elapsed_time)
                )
                elapsed_time += self.POLLING_INTERVAL

            # If we got here, we execeeded the polling inteval
            # Check for status one last time
            self._validate_success()

        def _validate_success(self):
            info = self.get_info()
            status = info.status
            if status == TrainingStatus.COMPLETED:
                return
            if status in [TrainingStatus.ERRORED, TrainingStatus.CANCELED]:
                raise RuntimeError(
                    f"Training process died with status {status} and message {info.errors}"
                )

        def load(self) -> ModuleBase:
            """A model future must be loadable with no additional arguments"""
            self.wait()

            # If we exit waiting without it throwing an exception,
            # we assume the job succeeded
            result = caikit.load(self.save_path)
            return result

    @staticmethod
    def _obj_to_txt(obj):
        message_bytes = pickle.dumps(obj)
        base64_bytes = base64.b64encode(message_bytes)
        txt = base64_bytes.decode("ascii")
        return txt

    def train(
        self,
        module_class: Type[ModuleBase],
        *args,
        save_path: Optional[str] = None,
        save_with_id: bool = False,
        num_gpus: int = None,
        num_cpus: int = None,
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

            *args
                Positional arguments for the train method

            save_path: str (Optional)
                Location on disk of where to save the model

            save_with_id: str (Optional)
                Whether to embed the training ID into the save path

            num_gpus: int (Optional)
                The number of gpus to be used for the training task

            num_cpus: int (Optional)
                The number of cpus to be used for the training task

            **kwargs
                Named arguments for the train method

        Returns:
            RayTrainModelFuture
        """

        ray_job_client = self.get_client()

        env_vars = {"module_class": self._obj_to_txt(module_class)}

        metadata = {}
        if save_path is not None:
            error.type_check("<RYT69631537E>", str, save_path=save_path)
            env_vars["save_path"] = save_path
            metadata["save_path"] = save_path

        # Validate number of CPUs and GPUs requested.
        # TODO: Should we have configurable limits on number of each that can be requested?
        default_resources = self.config.get("default_resources")
        if num_cpus is None and default_resources is not None:
            num_cpus = default_resources.get("cpu")
        if num_cpus is not None:
            error.type_check("<RYT19121015E>", int, num_cpus=num_cpus)
            error.value_check("<RYT81418146E>", num_cpus > 0)
            env_vars["requested_cpus"] = num_cpus

        if num_gpus is None and default_resources is not None:
            num_gpus = default_resources.get("gpu")
        if num_gpus is not None:
            error.type_check("<RYT94930817E>", int, num_gpus=num_gpus)
            error.value_check("<RYT87231812E>", num_gpus > 0)
            env_vars["requested_gpus"] = num_gpus

        # Serialize **kwargs and add them to environment variables
        my_kwargs = {}
        for key, value in kwargs.items():
            my_kwargs[key] = self._obj_to_txt(value)
        env_vars["kwargs"] = my_kwargs

        # Serialize *args and add them to environment variables
        my_args = []
        for arg in args:
            my_args.append(self._obj_to_txt(arg))
        env_vars["args"] = my_args

        if save_with_id:
            metadata["save_with_id"] = str(save_with_id)

        job_id = ray_job_client.submit_job(
            entrypoint="ray_submitter", runtime_env=env_vars, metadata=metadata
        )

        model_future = self.RayTrainModelFuture(self, job_id, save_with_id, save_path)

        return model_future

    def get_model_future(self, training_id: str) -> RayTrainModelFuture:
        """All shared train backends must be able to retrieve the future for a
        given training by id
        """

        if self.RayTrainModelFuture.ID_DELIMITER in training_id:
            ray_job_id = training_id.split(self.RayTrainModelFuture.ID_DELIMITER)[1]
        else:
            ray_job_id = training_id

        client = self.get_client()
        try:
            job_info = client.get_job_info(ray_job_id)
        except RuntimeError as e:
            logger.error(
                "Unable to retrieve info for Ray job ID [%s] with error [%s]",
                training_id,
                e,
            )
            return None

        # If we do not receive an exception, then it is a valid Ray job
        metadata = job_info.metadata
        # Metadata is stored as strings, so convert string to bool
        save_with_id = bool(metadata.get("save_with_id"))
        save_path = metadata.get("save_path")
        model_future = self.RayTrainModelFuture(
            ray_be=self,
            ray_job_id=ray_job_id,
            save_with_id=save_with_id,
            save_path=save_path,
        )
        return model_future


model_trainer_factory.register(RayJobTrainModule)
