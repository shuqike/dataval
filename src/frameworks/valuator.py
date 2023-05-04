class Valuator:
    pass


class StaticValuator(Valuator):
    def run(self):
        raise NotImplementedError


class DynamicValuator(Valuator):
    def one_step(self):
        raise NotImplementedError
