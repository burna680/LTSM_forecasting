
class Page:
    def __init__(self, name):
        self.name = name

    def render(self):
        raise NotImplementedError
