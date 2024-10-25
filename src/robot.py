
from example import Example

# Gonna be in some sort of loop because we are constantly running
def main():
    while True:
        ex = Example('Test')

        # Might have to run scripts async
        ex.example()

if __name__ == '__main__':
    main()