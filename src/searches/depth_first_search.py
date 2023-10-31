class DepthFirstSearch():
    def __init__(
        self,
    ) -> None:
        self.cache = {
            'visited': {},
            'queue': []
        }

    def get_value(self, location):
        previous_key = location[:-1]
        if previous_key in self.keys():
            previous = self.cache['visited'][previous_key]
            getattr(previous, self.get_next)()

    