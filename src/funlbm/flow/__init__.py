from .base import Flow, FlowConfig
from .flow3d import FlowD3, FlowD3Q13, FlowD3Q19

__all__ = ["Flow", "FlowD3", "FlowD3Q19", "FlowD3Q13", "FlowConfig", "create_flow"]


def create_flow(flow_config: FlowConfig, *args, **kwargs):
    return FlowD3(config=flow_config, *args, **kwargs)
