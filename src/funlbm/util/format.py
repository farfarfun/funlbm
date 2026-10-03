def tensor_format(data) -> list[float]:
    """把一组标量张量转换成可 JSON 序列化的 Python float 列表。

    Args:
        data: 可迭代对象，元素为位于任意 device 上的 0 维（标量）张量。

    Returns:
        依次对应 `data` 中每个元素的 Python float 列表。
    """
    return [float(i.cpu().numpy()) for i in data]
