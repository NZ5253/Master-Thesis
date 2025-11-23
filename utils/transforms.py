import math
def wrap_angle(a):
    return (a + math.pi)%(2*math.pi) - math.pi
