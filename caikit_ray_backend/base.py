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

"""Base class to initialize backend
"""

# Standard
from enum import Enum
from typing import Optional
import abc

# First Party
from caikit.core.module_backends import BackendBase


class SharedTrainBackendBase(BackendBase, abc.ABC):
    """Interface for a backend that can perform train on any given module

    A Shared backend is one that treats the given module as a black box and
    delegates the execution of that module's functionality to an alternate
    execution engine.
    """

    class TrainingStatus(Enum):
        QUEUED = 1
        RUNNING = 2
        COMPLETED = 3
        ERRORED = 4

    class ModelFuture(abc.ABC):
        @property
        @abc.abstractmethod
        def id(self) -> str:
            """Every model future must have a unique ID that can be used to look
            up the in-flight training
            """

        @property
        @abc.abstractmethod
        def save_path(self) -> Optional[str]:
            """If created with a save path, the future must expose it"""

        @abc.abstractmethod
        def get_status(self) -> "TrainingStatus":
            """Every model future must be able to poll the status of the
            training job
            """

        @abc.abstractmethod
        def cancel(self):
            """Terminate the given training"""

        @abc.abstractmethod
        def wait(self):
            """Block until the job reaches a terminal state"""

        @abc.abstractmethod
        def load(self):
            """A model future must be loadable with no additional arguments"""

    @abc.abstractmethod
    def train(
        self,
        module_class,
        *args,
        save_path: Optional[str] = None,
        **kwargs,
    ) -> ModelFuture:
        """Perform the given module's train operation and return the trained
        module instance.

        TODO: The return type here might be problematic in the case where the
            server performing the train operation is just a proxy for both train
            and inference. Consider some kind of lazy load proxy that would not
            require the model to be held in memory.

        Args:
            module_class (Type[ModuleBase]): The module class to train
            *args: Additional positional args to pass through to training
            save_path (Optional[str]): A path for saving the output model
            **kwargs: The args to pass through to training

        Returns:
            model_future (ModelFuture): A ModelFuture object for this backend
                that holds information about the in-progress training job
        """

    @abc.abstractmethod
    def get_future(self, training_id: str) -> Optional[ModelFuture]:
        """All shared train backends must be able to retrieve the future for a
        given training by id
        """
