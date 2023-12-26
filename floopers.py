def identity(input):
    return input


class Looper:
    def __init__(self, func_list):
        self.func_list = func_list

    def __loop(self, input, args):
        assert len(self.func_list) == len(
            input
        ), "input size should be equal to functions list"

        for var, func in zip(input, self.func_list):
            if func != identity:
                self.apply(func, var, args)

    def apply(self, input, func, args):
        pass


class FuncTrain(Looper):
    def __call__(self, input, args=None):
        self.results = []
        self.__loop(input=input)
        return self.result

    def apply(self, input, func):
        self.results.append(func(input))


class Processor(Looper):
    def apply(self, input, func, args):
        func(input, args)

    def __call__(self, input, args=None):
        self.__loop(input=input)
