def closest_power_of_2(n):
    r = 1
    while r * 2 <= n:
        r *= 2
    return r


if __name__ == '__main__':
    for i in range(20):
        print(i, closest_power_of_2(i))