class ParentClass:
    def my_method(self):
        print("This is the parent class method.")

class ChildClass(ParentClass):
    def my_method(self):
        print("This is the child class method.")

# create an instance of the child class
child = ChildClass()

# call the overridden method
child.my_method()