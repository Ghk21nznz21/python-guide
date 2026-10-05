''' Script Description '''

# Modules (File or library with python code, that can be imported) 
# and Packages (Collection of modules, related to each other)
import math  # Importing a module
from datetime import time  # Importing a module from a package


class Person:
    ''' class description ... '''
        
    # Class Variable 
    employes = 10
    
    # Class Constructor: used to initialize the instance variables
    def __init__(self, name: str, age: int):
        # Instance variables 
        self.name = name  
        self.age = age
        employes = 10


    def instanceMethod(self):
        ''' First param = self = poiter to the class instance '''
        return  self.age + 1 
    
    @classmethod  # Decorator -> Wrap a Function, modifying its behavior
    def classMethod(cls):
        ''' First param = self = poiter to the class instance
        Shared by all instances 
        Can be accessed directly without creating an instance 
        '''
        return cls.employes

    @staticmethod 
    def staticMethod(number: int = 10):
        '''
        Utility functions
        Don’t use any instance of class reference 
        '''
        return number ** 2


# Encapsulation: Apply restriction on access and modification
class BankAccount:
    def __init__(self, balance):
        self._balance = balance  # Protected variable
        self.__account_number = 12345  # Private variable


# Inheritance
class Student(Person, BankAccount):
    '''
    This is a child class.
    Auto imports every method and attributes of the parent classes
    It can override them 
    
    When using dot notation. It first checks on the child class and if not 
    found will look at the parent 

    '''
    def __init__(self, name, age, balance, student_id, ):
        # Super fetches those variables from parent 
        super(Person).__init__(name, age) # specify parent to fetch vars
        BankAccount().__init__(balance)  # another way of using it 
        self.student_id = student_id

    def myId(self):
        def localScopeFunction():
            return 'No one outside this cant access me'
        localScopeFunction()
        return f"Hello, I'm student {self.student_id}"


# Iteration, Iterator, Iterable
class MyIterator:
    '''
    Dunder methods __dunderMethodName__ : allow to interact and override 
        built-in functions and operators of the language
    
    Iterator vs Iterable:
        Iterable:
            object capable of returning its elements one at a time.
            It has an __iter__() method, which, when called, returns an iterator
            You can iterate over an iterable multiple times.
            Ex: lists, tuples, dictionaries, and strings
            
        Iterator --> object which iterates
            object that keeps track of its internal state and provides a way to 
                retrieve elements one by one
            You can only iterate over an iterator once
            It has both __iter__() and __next__() methods. And calls 
                next() until it raises StopIteration.
    '''
    def __init__(self, data):
        self.data = data
        self.index = 0

    def __iter__(self):
        ''' 
        Any objec with this is an Iterable, which means it can be iterated over several times, such as list()s and tuple()s.'''
        return self

    def __next__(self):
        # Class Constructor --> called when an object is created
        if self.index < len(self.data):
            result = self.data[self.index]
            self.index += 1
            return result
        else:
            raise StopIteration


# Decorator - Wrapper
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Something is happening before the function is called.")
        result = func(*args, **kwargs)
        print("Something is happening after the function is called.")
        return result
    return wrapper


def comparing():
    A, B = 2, 2
    print(A == B) # compares values 
    print(A is B) # compares IDs  
    print(A) # calls __bool__ and __len__ (Is False when  → False, None, Empty)
    print(A is None) # None has its own ID (Is False when  → None)

    print(isinstance(A, list)) # Compares type of the object. Accounts for inherance
    print(type(A) == list) # Compares type of the object.

if __name__ == "__main__":
    # class Method
    Person().classMethod()
    
    # instance method 
    person = Person("Alice", 30)
    print(person.instanceMethod())

    # Iteration, Iterator, Iterable
    my_data = [1, 2, 3, 4, 5]
    my_iterator = MyIterator(my_data)
    for item in my_iterator:
        print(item)


