#!/usr/bin/env python3


class Animal:
    def __init__(self, color, breed, age):
        self.color = color
        self.breed = breed
        self.age = age
        self.greeting = None

    def greet(self):
        print(self.greeting)


class Cat(Animal):
    def __init__(self, *args, **kwargs):
        super(Cat).__init__(*args, **kwargs)
        self.greeting = "meow"


class Dog(Animal):
    def __init__(self, *args, **kwargs):
        super(Cat).__init__(*args, **kwargs)
        self.greeting = "bork"


class Hamster(Animal):
    def __init__(self, *args, **kwargs):
        super(Cat).__init__(*args, **kwargs)
        self.greeting = "**bito**"
