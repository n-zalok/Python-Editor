class Cat():
    def __init__(self, Name):
        self.Name = Name
    
    def MeowNow(self, times):
        for _ in range(times):
            print(f"{self.Name} says meow!")