from .base import FlowBase, FlowConfig
from .d3 import FlowD3, FlowD3Q13, FlowD3Q15, FlowD3Q19, FlowD3Q27

__all__ = [
    "FlowBase",
    "FlowConfig",
    "FlowD3",
    "FlowD3Q13",
    "FlowD3Q15",
    "FlowD3Q19",
    "FlowD3Q27",
    "create_flow",
]


def create_flow(flow_config: FlowConfig, *args: object, **kwargs: object) -> FlowBase:
    """根据流场配置创建对应的流场对象。

    Args:
        flow_config: 流场配置，`param_type` 字段决定创建哪种离散速度模型，
            支持 "D3Q27"/"D3Q19"/"D3Q15"/"D3Q13"。

    Returns:
        FlowBase: 对应离散速度模型的流场对象。

    Raises:
        ValueError: 当 `flow_config.param_type` 不受支持时。
    """
    if flow_config.param_type == "D3Q27":
        return FlowD3Q27(config=flow_config, *args, **kwargs)
    elif flow_config.param_type == "D3Q19":
        return FlowD3Q19(config=flow_config, *args, **kwargs)
    elif flow_config.param_type == "D3Q15":
        return FlowD3Q15(config=flow_config, *args, **kwargs)
    elif flow_config.param_type == "D3Q13":
        return FlowD3Q13(config=flow_config, *args, **kwargs)
    else:
        raise ValueError(f"Unknown parameter type: {flow_config.param_type}")
