#!/usr/bin/env python
import unittest
import math

def fequal(a: float, b:float) -> bool:
    """ fuzzy equality for float, e is max rounding err"""
    e = 0.00001
    return  abs(a - b) < e

class Number:
    """ uses fequal for comparison"""
    def __init__(self, val):
        self.value = val
    def __repr__(self):
        return f"{self.value:f}"

    def __eq__(self, o):
        if isinstance(o, Number):
            return fequal(self.value, o.value)
        assert type(o) == int or type(o) == float
        return fequal(self.value, o)

    def __lt__(self, o):
        if isinstance(o, Number):
            return self.value < o.value
        return self.value < o
    def __sub__(self, o):
        if isinstance(o, Number):
            return self.value - o.value
        else:
            return self.value - o

    def __add__(self, o):
        if isinstance(o, Number):
            return self.value + o.value
        else:
            return self.value + o
    def __mul__(self, o):
        if isinstance(o, Number):
            return self.value * o.value
        else:
            return self.value * o
    def __truediv__(self, o):
        if isinstance(o, Number):
            return self.value / o.value
        else:
            return self.value / o
    def __pow__(self, o):
        if isinstance(o, Number):
            return self.value ** o.value
        else:
            return self.value ** o

class Point:
    def __init__(self, x,y,z):
        self.x = Number(x)
        self.y = Number(y)
        self.z = Number(z) 

    def __add__(self, other):
        x,y,z = 0,0,0
        x = self.x + other.x
        y = self.y + other.y
        z = self.z + other.z
        return Point(x,y,z)
       
    def __sub__(self, other):
        x,y,z = 0,0,0
        x = self.x - other.x
        y = self.y - other.y
        z = self.z - other.z
        return Point(x,y,z)
       
    def __eq__(self, other):
        assert isinstance(other, Point)
        return fequal(self.x, other.x) \
            and fequal(self.y, other.y)\
            and fequal(self.z, other.z)

    def __lt__(self, o):
        return self.x < o.x and self.y < o.y and self.z < o.z
    def __repr__(self):
        return f"Point({self.x},{self.y},{self.z})"

class Vector(Point):
    def __repr__(self):
        return f"Vector({self.x},{self.y},{self.z})"

    def __mul__(self, scalar):
        assert type(scalar) == int or type(scalar) == float
        return Vector(self.x*scalar, self.y*scalar, self.z*scalar)

    def __div__(self, scalar):
        assert type(scalar) == int or type(scalar) == float
        return Vector(self.x/scalar, self.y/scalar, self.z/scalar)

    def length(self) -> float:
        """ magnitude/length """
        return Number(math.sqrt(self.x**2 + self.y**2 + self.z**2))
    def normalize(self):
        """ return normalized vector from self"""
        mag = self.length()
        return Vector(self.x / mag, self.y / mag, self.z / mag)
    def dot(self, other):
        """ scalarproduct/dot product """
        assert isinstance(other, Vector)
        v =  self.x * other.x + self.y * other.y + self.z * other.z
        return Number(v)

import tkinter as Tk
class Canvas:
    def __init__(self, width, height):
        self.root = Tk.Tk()
        self.C = Tk.Canvas(self.root, width=width, height=height)
        self.C.pack()

    def stop(self):
        self.root.quit()

    def run(self):
        self.root.mainloop()


class Tests(unittest.TestCase):
    def test_point_add(self):
        l = Point(1,2,3)
        r = Point(3,4,5)
        l += r
        self.assertEqual(l, Point(4,6,8))

    def test_point_sub(self):
        l = Point(-1,-33,-7)
        r = Point(21, 42, 63)
        r -= l
        self.assertEqual(r, Point(22, 75, 70)) 

    def test_vector_add(self):
        self.assertEqual(Vector(0,-1,0) + Vector(1, 2, 1), Vector(1,1,1))
    def test_vector_mul(self):
        self.assertEqual(Vector(0,0,0) * 10, Vector(0,0,0))
        self.assertEqual(Vector(9,1,1) * 5, Vector(45,5,5))
        self.assertEqual(Vector(9,1,1) * 0.5, Vector(4.5,0.5,0.5))
    def test_vector_len(self):
        self.assertEqual(Vector(1,0,0).length(), 1)
        self.assertEqual(Vector(0,1,0).length(), 1)
        self.assertEqual(Vector(0,0,1).length(), 1)

        self.assertEqual(Vector(10,0,0).length(), 10)
        self.assertEqual(Vector(11.313708,9.797958,5.656854).length(), 16.0)
    def test_vector_norm(self):
        self.assertEqual(Vector(40,0,0).normalize(), Vector(1,0,0))
        self.assertLess(Vector(1234, 998, 0x234).normalize(), Vector(1,1,1))

    def test_vector_dot(self):
        #orthogonal
        self.assertEqual(Vector(1,0,0).dot(Vector(0,1,0)), 0)
        self.assertEqual(Vector(0,0,1).dot(Vector(0,1,0)), 0)
        # directions
        self.assertEqual(Vector(0,0,1).dot(Vector(0,0,1)), 1)
        self.assertEqual(Vector(0,0,-1).dot(Vector(0,0,1)), -1)



import argparse as Ap
import sys

if __name__ == "__main__":
    ap = Ap.ArgumentParser("Mondlander")
    ap.add_argument("--test", "-T", help="Run unit tests", action="store_true")
    args = ap.parse_args()

    if args.test:
        unittest.main(argv=[sys.argv[0], "--verbose"])
    else:
        c = Canvas(500,300)
        c.run()
