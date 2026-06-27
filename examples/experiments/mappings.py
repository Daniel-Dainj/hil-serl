from importlib import import_module


def _make_config_factory(module_name):
    def factory():
        return getattr(import_module(module_name), "TrainConfig")()

    return factory


CONFIG_MAPPING = {
    "ram_insertion": _make_config_factory("experiments.ram_insertion.config"),
    "usb_pickup_insertion": _make_config_factory("experiments.usb_pickup_insertion.config"),
    "workpiece_pickup": _make_config_factory("experiments.workpiece_pickup.config"),
    "object_handover": _make_config_factory("experiments.object_handover.config"),
    "egg_flip": _make_config_factory("experiments.egg_flip.config"),
}
