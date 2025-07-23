from abc import ABC, abstractmethod


class Validator(ABC):

    def __set_name__(self, owner, name):
        self.private_name = "_" + name

    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private_name)

    def __set__(self, obj, value):
        self.validate(value)
        setattr(obj, self.private_name, value)

    @abstractmethod
    def validate(self, value):
        pass


class OneOf(Validator):

    def __init__(self, *options):
        self.options = set(options)

    def validate(self, value):
        if value not in self.options:
            raise ValueError(f"Expected value to be on of {self.options}")


class Number(Validator):

    def __init__(self, minvalue=None, maxvalue=None):
        self.minvalue = minvalue
        self.maxvalue = maxvalue

    def validate(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError(f"Expected value to be an int or float")

        if self.minvalue is not None and value < self.minvalue:
            raise ValueError(f"Expected value to be at least {self.minvalue}")
        if self.maxvalue is not None and value > self.maxvalue:
            raise ValueError(f"Expected value to be no more than {self.maxvalue}")


class String(Validator):

    def __init__(self, minsize=None, maxsize=None, predicate=None):
        self.minsize = minsize
        self.maxsize = maxsize
        self.predicate = predicate

    def validate(self, value):
        if not isinstance(value, str):
            raise TypeError(f"Expected value to be a string")
        if self.minsize is not None and len(value) < self.minsize:
            raise ValueError(f"Expected size to be at least {self.minsize}")
        if self.maxsize is not None and len(value) > self.maxsize:
            raise ValueError(f"Expected size to be no more than {self.maxsize}")


if __name__ == "__main__":

    class Student:

        name = String(0, 20)

        math = Number(0, 150)
        physics = Number(0, 100)

        def __init__(self, name, math_score, physics_score):
            self.name = name
            self.math = math_score
            self.physics = physics_score

        def show_score(self):
            print(
                f"Name: {self.name}, Math Score: {self.math}, Physics Score: {self.physics}"
            )

        def update_score(self, math_score, physics_score):
            self.math = math_score
            self.physics = physics_score

    John = Student("John", 60, 80)
    John.show_score()
    John.math = 130
    John.physics = 100
    John.show_score()

    Mary = Student("Mary", 180, 80)
