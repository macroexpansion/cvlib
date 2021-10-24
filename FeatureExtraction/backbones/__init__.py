from .iresnet import iresnet50, iresnet100


def get_model(name, **kwargs):
    if name == "r50":
        return iresnet50(False, **kwargs)
    elif name == "r100":
        return iresnet100(False, **kwargs)
    else:
        raise ValueError()
