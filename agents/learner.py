import torch as tc
from torch import nn

from .replay_buffer import ReplayBuffer

from pathlib import Path
from abc import abstractmethod, ABC

from numpy.typing import NDArray
from torch.types import Device
from torch import Tensor
from typing import Any, Self


class Learner(nn.Module, ABC):
    '''Interface for all my different agent versions'''

    @abstractmethod
    def act(self, state: Tensor, actions: Tensor) -> NDArray:
        '''Choose an action from `actions` based on the state. `state` may be singular or batched'''

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        '''Convert model to dictionary'''

    @classmethod
    @abstractmethod
    def from_dict(cls, model_dict: dict[str, Any], device: Device) -> Self:
        '''Load model from dictionary created by `self.to_dict`'''

    @abstractmethod
    def mlflow_get_sample_weights(self) -> dict[str, float]:
        '''Return any sample weights for logging, preferably one from each separate model part'''

    @property
    @abstractmethod
    def active_submodels(self) -> list[nn.Module]:
        '''Return a list of contained models that have a corresponding target model. Used for target updating.'''

    @property
    @abstractmethod
    def target_submodels(self) -> list[nn.Module]:
        '''Return a list of target models in the same order as `active_submodels`. Used for target updating.'''

    @abstractmethod
    def training_session(
        self,
        replay_buffer: ReplayBuffer,
        n_epochs: int,
        batch_size: int,
        lr: float,
        gamma: float,
    ) -> dict[str, float]:
        '''Perform a training step and return training metrics'''


    def set_device(self, device: Device):
        self.to(device)
        self.device = device

        return self


    def save(self, path_to_dir: str | Path, epoch: int):
        '''Save model to `<path_to_dir>/<epoch>.pth`'''
        # make sure path is of type Path
        if not isinstance(path_to_dir, Path):
            path_to_dir = Path(path_to_dir)

        # make sure path points to a directory
        assert path_to_dir.suffix == "", "path_to_dir must point to a directory!"

        # create folder if necessary
        path_to_dir.mkdir(parents=True, exist_ok=True)

        # get model dictionary
        model_dict = self.to_dict()

        # save dictionary to file
        tc.save(model_dict, path_to_dir / f"epoch_{epoch}.pth")


    @classmethod
    def load(cls, path_to_dir: str | Path, epoch: int, device: Device) -> Self:
        '''Load model from `<path_to_dir>/<epoch>.pth`'''
        # make sure path is of type Path
        if not isinstance(path_to_dir, Path):
            path_to_dir = Path(path_to_dir)

        # make sure path points to a directory
        assert path_to_dir.suffix == "", "path_to_dir must point to a directory!"

        # load model dictionary
        model_dict = tc.load(path_to_dir / f"epoch_{epoch}.pth", map_location=device) # type: ignore

        # create model from dictionary
        out = cls.from_dict(model_dict, device)

        return out


    def target_update_polyak(self, rho: float) -> None:
        for q, q_target in zip(self.active_submodels, self.target_submodels):
            v = tc.nn.utils.parameters_to_vector(q.parameters())
            v_target = tc.nn.utils.parameters_to_vector(q_target.parameters())

            v_new = rho * v_target + (1 - rho) * v

            tc.nn.utils.vector_to_parameters(v_new, q_target.parameters())


    def target_update_copy(self) -> None:
        for q, q_target in zip(self.active_submodels, self.target_submodels):
            v = tc.nn.utils.parameters_to_vector(q.parameters())
            tc.nn.utils.vector_to_parameters(v, q_target.parameters())
