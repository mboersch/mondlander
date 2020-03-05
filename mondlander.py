#!/usr/bin/env python3
import unittest
import math

def sin(deg):
    """ degrees (360deg = 2pi rad) """
    return round(math.sin(math.radians(deg)), 15)
def cos(deg):
    return round(math.cos(math.radians(deg)), 15)
pi = math.pi

def fequal(a: float, b:float) -> bool:
    """ fuzzy equality for float, e is max rounding err"""
    e = 0.00001
    return  abs(a - b) < e

class Number(float):
    """ uses fequal for comparison"""
    def __init__(self, val):
        return super().__init__()

    def __eq__(self, o):
        return fequal(self, o)


class Point:
    def __init__(self, x,y,z=0):
        """ should accept tuple"""
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
      
    def __mul__(self, scalar):
        assert isinstance(scalar, int)\
                or isinstance(scalar, float) \
                or isinstance(scalar, Number)
        x = self.x * scalar
        y = self.y * scalar
        z = self.z * scalar
        return Point(x,y,z)

    def __truediv__(self, scalar):
        assert isinstance(scalar, int)\
                or isinstance(scalar, float) \
                or isinstance(scalar, Number)
        x = self.x / scalar
        y = self.y / scalar
        z = self.z / scalar
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

def _mkempty(dim):
    def n0():
        for i in range(dim):
            yield 0
    return list(n0())
class Matrix:
    """ Knock, knock, Neo"""

    def __init__(self, *list_of_rows):
        """ accepts list of rows, length of rows must be same"""
        assert len(list_of_rows) > 0
        dim0 = len(list_of_rows[0])
        self._rows=[]
        for row in list_of_rows:
            assert len(row) == dim0
            self._rows.append(list(row))

    def cols(self):
        return len(self._rows[0])

    def rows(self):
        return len(self._rows)

    def __repr__(self):
        return f"Matrix({self._rows})"

    def __mul__(self, other):
        assert isinstance(other, Matrix) or isinstance(other, Vector)
        if isinstance(other, Vector):
            """ turn vector into column matrix"""
            assert self.rows() == 3 and self.cols() == 3
            other = Matrix([other.x], [other.y], [other.z])
        #print(f"{self.rows()}x{self.cols()} * {other.rows()}x{other.cols()}")
        res = []
        for i in range(0, self.rows()):
            res.append([])
            for j in range(0, other.cols()):
                tmp = 0
                for k in range(0, other.rows()):
                    tmp += self._rows[i][k] * other._rows[k][j]
                res[i].append(tmp)
        return Matrix(*res)

    def __iter__(self):
        for r in range(0, self.rows()):
            for c in range(0, self.cols()):
                yield (r,c)

    def __getitem__(self, ij):
        """ indexing by tuples .... """
        i,j = ij
        return self._rows[i][j]

    def __eq__(self, other):
        ok = (self.rows() == other.rows()) and (self.cols() == other.cols())
        if not ok: 
            return False
        for i,j in self:
            if not fequal(self[i,j], other[i,j]):
                return False
        return True


class RotZ(Matrix):
    """ Z-axis rotation"""
    def __init__(self, degrees):
        super().__init__( 
            [cos(degrees), -1*sin(degrees), 0],
            [sin(degrees), cos(degrees), 0],
            [0, 0, 1]
        )

import tkinter as Tk
class Canvas:
    def __init__(self, width, height):
        self.root = Tk.Tk()
        self.C = Tk.Canvas(self.root, width=width, height=height)
        self.C.pack()

    def width(self):
        return self.width

    def height(self):
        return self.height

    def stop(self):
        self.root.quit()

    def draw_polygon(self, tag, offsetxy,  mesh):
        """ convert list of Points() into polygon on canvas,
        starting at offsetxy in the canvas space. """
        self.delete(tag) #cleanup previous call
        xycords = []
        for node in mesh:
            xycords.append( (node.x + offsetxy.x , node.y + offsetxy.y) )
        return self.C.create_polygon(xycords, width=1, tag=tag)

    def delete(self, id_or_name):
        self.C.delete(id_or_name)

    def run(self):
        self.root.mainloop()

# drawn with pen and paper, create_line on canvas:
#def LANDER = [5,2, 3,6,  5,6, 4,9, 3,9, 6,9, 4,9, 5,6, 10,6, 11,9, 12,9, 9,9,
#11,9 , 10,6, 12,6, 10,2, 5,2]


def scale(orig_mesh,  factor):
    """ scale by  factor """
    mesh=[]
    for point in orig_mesh:
        mesh.append(point * factor)
    return mesh

def rotate(orig_mesh, degree):
    """ rotate by degrees, around z axis """
    rm = RotZ(degree)
    mesh=[]
    for point in orig_mesh:
        rot = rm * Vector(point.x, point.y, point.z)
        mesh.append(Point(rot[0,0], rot[1,0], rot[2,0]))
    return mesh

def translate(orig_mesh, whereto):
    """ translate position to Vector whereto """
    mesh=[]
    for point in mesh:
        mesh.append(point + whereto)
    return mesh

class Lander:
    LANDER = [(5,2), (3,6), (5,6), (4,9), (3,9), (3,10),  (6,10), (6,9), (5,9),
        (6,6), (10,6), (11,9), (12,9), (12,10), (9,10), (9,9), (10,9) , (9,6),
        (12,6), (10,2), (5,2)]
    def __init__(self, x,y):
        """ create lander at x/y coords"""
        self.coords = Point(x,y)
        self.canvas_id = None
        self.mesh = []
        for xy in self.LANDER:
            self.mesh.append(Point(*xy))
        #keep deep copy,
        self.orig_mesh = list(self.mesh)

    def reset(self):
        self.mesh = list(self.orig_mesh)

    def position(self):
        return self.coords

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
    def test_matrix(self):

        with self.assertRaises(AssertionError):
            m = Matrix([1,2,3],[1,2])

        m = Matrix([1,0,0], [0,1,0], [0,0,1])
        self.assertEqual(m._rows, [[1,0,0],[0,1,0],[0,0,1]])

        self.assertNotEqual(Matrix([123,456],[123,456]), 
                Matrix([-1234,234],[1,2]))

    def test_matrix_mul(self):
        v = Vector(1,1,1)
        m = Matrix([1,2,3],[3,4,5],[5,6,7])
        mv = m * v
        self.assertEqual(mv, Matrix([6],[12], [18]))
        self.assertEqual(mv.rows(), 3)
        self.assertEqual(mv.cols(), 1)

    def test_matrix_rotate(self):
        r90 = RotZ(90)
        self.assertEqual(r90, Matrix([0, -1, 0],[1,0,0],[0,0,1]))
        r180 = RotZ(180)
        self.assertEqual(r180, Matrix([-1, 0, 0],[0,-1,0],[0,0,1]))
        r270 = RotZ(270)
        self.assertEqual(r270, Matrix([0, 1, 0],[-1,0,0],[0,0,1]))

        orig = Matrix([1,2,3],[4,5,6],[7,8,9])
        idem = orig * RotZ(360)
        self.assertEqual(idem, orig)

def drawLanderTest():
    c = Canvas(500,300)
    l = Lander(250,150)
    def scale_animation(i, up, obj):
        if i > 8:
            up = -1
        elif i < 0.8:
            up = +1
        mesh = rotate(scale(obj.mesh, i), 1 + i*10)
        c.draw_polygon("Lander", l.position(), mesh)

        c.C.after(33, scale_animation,  i + 0.1 *up, up, obj)
    scale_animation(1, 1, l)
    c.run()

import argparse as Ap
import sys

if __name__ == "__main__":
    ap = Ap.ArgumentParser("Mondlander")
    ap.add_argument("--test", "-T", help="Run unit tests", action="store_true")
    args = ap.parse_args()

    if args.test:
        unittest.main(argv=[sys.argv[0], "--verbose"])
    else:
        drawLanderTest()
