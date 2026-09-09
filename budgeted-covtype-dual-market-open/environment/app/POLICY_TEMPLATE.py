class Policy:
    def __init__(self, data_dir):
        raise NotImplementedError

    def select_queries(self, labeled, budget_left, prices):
        raise NotImplementedError

    def select_next(self, observed, budget_left, prices):
        raise NotImplementedError

    def predict(self, observed):
        raise NotImplementedError
