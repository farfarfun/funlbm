def tensor_format(data):
    return [float(i.cpu().numpy()) for i in data]
