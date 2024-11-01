
from example import Example
import asyncio
from datetime import datetime

async def test_script1():
    print("1 started")
    await asyncio.sleep(1)
    print("1 finished")

async def test_script2():
    print("2 started")
    await asyncio.sleep(2)
    print("2 finished")

async def test_script3():
    print("3 started")
    await asyncio.sleep(3)
    print("3 finished")

# Gonna be in some sort of loop because we are constantly running
async def main():
    while True:

        async with asyncio.TaskGroup() as tg:
            task1 = tg.create_task(
                test_script1()
            )
            task2 = tg.create_task(
                test_script2()
            )
            task3 = tg.create_task(
                test_script3()
            )
            print(datetime.now())

        print(datetime.now())

if __name__ == '__main__':
    asyncio.run(main())