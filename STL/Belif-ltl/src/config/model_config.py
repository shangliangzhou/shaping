from abc import abstractmethod, ABC
from typing import Type, Optional

from dataclasses import dataclass

from torch import nn

from model.env import ConvEnvNet, StandardEnvNet
from model.ltl.set_network import SetNetwork


class AbstractModelConfig(ABC):
    @abstractmethod
    def build(self, input_shape: tuple[int, ...]) -> nn.Module:
        pass


@dataclass
class StandardNetConfig(AbstractModelConfig):
    layers: list[int]
    activation: Optional[Type[nn.Module]]

    def build(self, input_shape: tuple[int,]) -> nn.Module:
        return StandardEnvNet(input_shape[0], self.layers, self.activation)


@dataclass
class ConvNetConfig(AbstractModelConfig):
    channels: list[int]
    kernel_size: tuple[int, int]
    activation: Type[nn.Module]

    def build(self, input_shape: tuple[int, int, int]) -> nn.Module:
        return ConvEnvNet(input_shape, self.channels, self.kernel_size, self.activation)


@dataclass
class SetNetConfig(AbstractModelConfig):
    layers: list[int]
    activation: Type[nn.Module]

    def build(self, input_shape: int) -> nn.Module:
        return SetNetwork(input_shape, self.layers, self.activation)


@dataclass
class ActorConfig:
    layers: list[int]
    activation: Optional[Type[nn.Module]] | dict[str, Type[nn.Module]]
    state_dependent_std: bool = False


@dataclass
class ModelConfig:
    actor: ActorConfig #定义策略网络（Actor）的配置
    critic: StandardNetConfig #定义价值网络（Critic）的配置
    ltl_embedding_dim: int #LTL嵌入的维度
    num_rnn_layers: int #RNN层数
    env_net: Optional[AbstractModelConfig] #定义环境网络（EnvNet）的配置,环境特征处理网络的配置，可以是StandardNetConfig或ConvNetConfig或None
    set_net: SetNetConfig #定义集合网络（SetNet）的配置,用于处理LTL公式中的集合操作


zones = ModelConfig(
    actor=ActorConfig(
        layers=[64, 64, 64],
        activation=dict(
            hidden=nn.ReLU,
            output=nn.Tanh
        ),
        state_dependent_std=True
    ),
    critic=StandardNetConfig(
        layers=[64, 64],
        activation=nn.Tanh
    ),
    ltl_embedding_dim=16,
    num_rnn_layers=1,
    env_net=StandardNetConfig(
        layers=[128, 64],
        activation=nn.Tanh
    ),
    set_net=SetNetConfig(
        layers=[32, 16],
        activation=nn.ReLU
    )
)

letter = ModelConfig(
    actor=ActorConfig(
        layers=[64, 64, 64],
        activation=nn.ReLU,
    ),
    critic=StandardNetConfig(
        layers=[64, 64],
        activation=nn.Tanh
    ),
    ltl_embedding_dim=32,
    num_rnn_layers=1,
    env_net=ConvNetConfig(
        channels=[16, 32, 64],
        kernel_size=(2, 2),
        activation=nn.ReLU
    ),
    set_net=SetNetConfig(
        layers=[32, 32],
        activation=nn.ReLU
    )
)

flatworld = ModelConfig(
    actor=ActorConfig(
        layers=[64, 64, 64],
        activation=nn.ReLU,
    ),
    critic=StandardNetConfig(
        layers=[64, 64],
        activation=nn.ReLU
    ),
    ltl_embedding_dim=16,
    num_rnn_layers=1,
    env_net=StandardNetConfig(
        layers=[16, 16],
        activation=nn.ReLU
    ),
    set_net=SetNetConfig(
        layers=[32, 16],
        activation=nn.ReLU
    )
)
