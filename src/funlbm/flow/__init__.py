from .base import Flow, FlowConfig
from .d3 import FlowD3, FlowD3Q13, FlowD3Q15, FlowD3Q19, FlowD3Q27

__all__ = [
    "Flow",
    "FlowD3",
    "FlowD3Q19",
    "FlowD3Q13",
    "FlowConfig",
    "create_flow",
    "FlowD3Q15",
    "FlowD3Q27",
]


def create_flow(flow_config: FlowConfig, *args, **kwargs):
    # return FlowD3(config=flow_config, *args, **kwargs)

    """解析3D模型参数

    根据参数类型创建对应的参数对象

    Args:
        flow_type: 参数类型,支持"D3Q27"/"D3Q19"/"D3Q15"/"D3Q13"

    Returns:
        Param: 参数对象

    Raises:
        ValueError: 当参数类型不支持时
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
        raise ValueError("Unknown parameter type: {}".format(flow_config.param_type))
