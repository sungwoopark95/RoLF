import numpy as np
import argparse

parser = argparse.ArgumentParser(description="A program to obtain prime numbers between two numbers", 
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--nums", "-N", type=int, nargs=2)


def check_prime(num:int) -> tuple:
    if num - int(num) > 0:
        raise ValueError(f"Input should be an integer, not a float with decimals")
    if num <= 0:
        return False, None
    if num == 1:
        return False, 1
    sqrt = np.sqrt(num)
    flag = True
    factors = []
    for i in range(2, int(sqrt)+1):
        if num % i == 0:
            flag = False
            factors.append(i)
    if flag:
        return True, None
    return False, factors


def get_primes(start:int, end:int):   
    primes = []
    for n in range(start, end+1):
        flag, _ = check_prime(n)
        if flag:
            primes.append(n)        
    return primes


if __name__ == "__main__":
    args = parser.parse_args()
    nums = args.nums
    if len(nums) > 2:
        raise ValueError(f"For the {args.mode} mode, you can only pass 2 inputs.")
    else:
        a, b = nums
    
    prime_list = [str(item) for item in get_primes(start=a, end=b)]
    print(",".join(prime_list))
