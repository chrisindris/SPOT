class Detection():
    def __init__(self, name='a') -> None:
        self.name = name


class THUMOSdetection(Detection):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
