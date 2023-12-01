def print_all_evens(array):
    for a in array:
        if a%2==0:
            print(a)



def print_pairs_with_sum(array, target):
    for a in array:
        for b in array:
            if a+b==target:
                print(a, b)


def print_pairs_with_sum(array, target):
    for left in range(len(array)):
        for right in range(left+1, len(array)):
            a= array[left]
            b= array[right]
            if a+b==target:
                print(a, b)



def print_even_sequences(array):
    for a in array:
        if a%2==0:
            print(f"Sequence for {a}")
            for k in range(0, a):
                print(k)



# O(n) ---> n= biggest prime number
def is_prime(n):
    if n<2:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True


# O(a*p)
def print_all_primes(array):
    for k in array:
        if is_prime(k):
            print(k)




def create_table(array):
    min_value= min(array)
    max_value= max(array)

    # create list of size (max-min +1)
    table= [0]*(max_value-min_value+1)
    return table




def count_pairs_which_sum_to_max(array):
    max_value= max(array)
    count=0

    for left in range(0, len(array)):
        for right in range(left+1, len(array)):
            left_value= array[left]
            right_value= array[right]
            if left_value+right_value==max_value:
                count+=1
    return count