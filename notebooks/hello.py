from os import keras

def get_name():
    return input("Enter your name: ")

def greet(name):
    print(f"Hello, {name}")

def main():
    name = get_name()
    greet(name)

if __name__ == "__main__":
    main()