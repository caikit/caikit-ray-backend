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
from typing import Type

# Third Party
import ray

# First Party
from caikit.core.modules import ModuleBase
from caikit.core.toolkit.errors import error_handler
import alog

log = alog.use_channel("RAYTRN")
error = error_handler.get(log)


@ray.remote(num_gpus=1, num_cpus=1)
class RayTrainingActor:
    """A RayTrainingActor is a class that can be instantiated as a Ray Actor
    to call the training functions of caikit modules
    """

    def __init__(self, module_class: Type[ModuleBase]):
        """Create a new RayTrainingActor that is mapped to one specific caikit module.
            If the module information passed in is invalid, an exception will be thrown
            on intialization.

        Args:
            module_class: str
                Fully qualified path of module class
                i.e. "caikit_example.modules.example_module.ExampleClass
                This module should extend ModuleBase and implement train() and save() methods
        """

        self.training_module = module_class

    def _import_mod(self, training_module):
        components = training_module.split(".")
        mod = __import__(components[0])
        for comp in components[1:]:
            mod = getattr(mod, comp)
        return mod

    def train_and_save(self, model_path, *args, **kwargs):
        """This is essentially a proxy to a caikit module's train and save methods
            This Ray actor will invoke the module's train and save functions locally

        Args:
            model_path: str
                Location on disk where the save method should save the model to

            *args
                Positional arguments for the train method

            **kwargs
                Named arguments for the train method
        """
        if model_path:
            error.type_check("<RYT24249644E>", str, model_path=model_path)

        log.debug("<RYT57616295D>", "Beginning training")

        model = self.training_module.train(*args, **kwargs)

        log.debug("<RYT45386862D>", "Training complete, beginning save")
        model.save(model_path)

        log.debug("<RYT39131219D>", "Save complete")
