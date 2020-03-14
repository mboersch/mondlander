#!/usr/bin/env python3
import unittest
import math
import enum
import random

def sin(deg):
    """ degrees (360deg = 2pi rad) """
    return round(math.sin(math.radians(deg)), 15)
def cos(deg):
    return round(math.cos(math.radians(deg)), 15)

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
        assert type(scalar) == int or type(scalar) == float or type(scalar)== Number
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


# some 3x3 matrixes:
class RotZ(Matrix):
    """ Z-axis rotation"""
    def __init__(self, degrees):
        super().__init__( 
            [cos(degrees), -1*sin(degrees), 0],
            [sin(degrees), cos(degrees), 0],
            [0, 0, 1]
        )
   
class Identity(Matrix):
    def __init__(self, scale=1):
        super().__init__(
            [1 * scale,0,0],
            [0,1 * scale,0],
            [0,0,1 * scale]
        )

import tkinter as Tk

class Action(enum.IntFlag):
    none = 0
    left = 1 << 1
    right = 1 << 2
    up =  1 << 3
    down = 1 << 4
    quit = 1 << 5

class UserInput:
    def __init__(self, widget):
        self.W=widget
        self.W.focus_set() #keyboard 
        self.W.bind_all('<KeyRelease>', self.on_event)
        self.W.bind_all('<KeyPress>', self.on_event)
        self.event = None
        self.action = Action.none

    def on_event(self, event):
        def set_action(what):
            if event.type == Tk.EventType.KeyPress:
                self.action |= what
            elif event.type == Tk.EventType.KeyRelease:
                self.action &= ~what
            else:
                raise "unexpected event!"

        if (event.keysym == "Escape") or event.keysym == "q":
            set_action(Action.quit)
        if (event.keysym == "space") or event.keysym == "w" :
            set_action(Action.up)
        if (event.keysym == "Left") or event.keysym == "a":
            set_action(Action.left)
        if (event.keysym == "Right") or event.keysym == "d":
            set_action(Action.right)
        if (event.keysym == "Down") or event.keysym == "s":
            set_action(Action.down)


    def print(self, *args):
        print(*args)

    def __str__(self):
        return f"UserInput<{self.event}>"

    def do(self):
        return self.action

class Canvas:
    #pseudo alpha channel via stipple pattern
    stipple_patterns = [12,25,50,75]

    def __init__(self, width, height):
        self.root = Tk.Tk()
        self.C = Tk.Canvas(self.root, width=width, height=height,
                highlightthickness=0)
        self.C.pack()
        self.user_input = UserInput(self.C)

    def background(self, color):
        self.C["background"] = color

    def width(self):
        return float(self.C["width"])

    def height(self):
        return float(self.C["height"])

    def stop(self):
        self.root.quit()

    def draw_polygon(self, tag, mesh, color='black'):
        """ convert list of Points() into polygon on canvas,
        starting at offsetxy in the canvas space. """
        self.delete(tag) #cleanup previous call
        xycords = []
        for node in mesh:
            xycords.append( (node.x, node.y) )
        return self.C.create_polygon(xycords, width=1, tag=tag, fill=color)

    def draw_text(self, tag, offsetxy, color, text):
        self.delete(tag)
        text_id = self.C.create_text(offsetxy.x, offsetxy.y,
                tag=tag, anchor="nw", fill=color, font='Helvetica 18 bold')
        self.C.itemconfig(text_id, text=text)
        return text_id

    def draw_oval(self, tag,  vw, xy, color, stipple):
        self.delete(tag)
        st=""
        if stipple > 0:
            what = self.stipple_patterns[stipple % len(self.stipple_patterns)]
            st=f"gray{what}"
        tag = self.C.create_oval(*vw, *xy, fill=color, stipple=st)
        return tag 

    def delete(self, id_or_name):
        self.C.delete(id_or_name)

    #event loop
    def after(self, ms_int, *args):
        self.C.after(ms_int, *args)

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
    isvector=False
    if isinstance(orig_mesh, Vector): #XXX hack
        orig_mesh = [orig_mesh]
        isvector=True

    rm = RotZ(degree)
    mesh=[]
    for point in orig_mesh:
        rot = rm * Vector(point.x, point.y, point.z)
        mesh.append(Point(rot[0,0], rot[1,0], rot[2,0]))

    if isvector:
        return mesh[0]
    return mesh

def translate(orig_mesh, whereto):
    """ translate position to Vector whereto """
    mesh=[]
    for point in orig_mesh:
        mesh.append(point + whereto)
    return mesh

class Lander:
    LANDER = [(5,2), (3,6), (5,6), (4,9), (3,9), (3,10),  (6,10), (6,9), (5,9),
        (6,6), (10,6), (11,9), (12,9), (12,10), (9,10), (9,9), (10,9) , (9,6),
        (12,6), (10,2), (5,2)]
    def __init__(self, x,y, color='silver'):
        """ create lander at x/y coords"""
        self.coords = Point(x,y)
        self.canvas_id = None
        self.velocity = Vector(0,0,0)
        self.fuel = Number(100)
        self.color = color
        self.mesh = []
        for xy in self.LANDER:
            self.mesh.append(Point(*xy))
        #keep deep copy
        self.orig_mesh = list(self.mesh)

    def reset(self):
        self.mesh = list(self.orig_mesh)

    def position(self):
        """ currenty x/y position in canvas space.
        TODO bounding box would be nice
        """
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

    def test_vector_comp(self):
        self.assertTrue( Vector(0,0,0) == Vector(0,0,0))
        #self.assertTrue( Vector(0,1,0) < Vector(0,2,0))
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
    def test_physics(self):
        #TODO throw something 45degrees, 10m/s. check max height, distance
        #wolfram says: 254.9cm max height, 10.2 x distnce
        pass


def animTest():
    """ scale and rotation """
    c = Canvas(500,300)
    l = Lander(250,150)
    def scale_animation(i, up, obj):
        if i > 8:
            up = -1
        elif i < 0.8:
            up = +1
        mesh = rotate(scale(obj.mesh, i), 1 + i*10)
        mesh = translate(mesh, l.position())
        c.draw_polygon("Lander", mesh)

        c.C.after(33, scale_animation,  i + 0.1 *up, up, obj)
    scale_animation(1, 1, l)
    c.run()
    
def altitude(obj, canvas):
    """ 0,0 is top left, height would be canvas.size.
    find the maximum point in obj, return difference to canva.size
    """
    mp = Point(0,0,0)
    for p in obj:
        if p.y > mp.y:
            mp.y = p.y
    return canvas.height() - mp.y


class Game:
    """ 
    TODO 
    - add view matrix, adjust view to viewport
    """
    def __step(self):
        self.userinput()
        if not self.done:
            self.step()
        self.canvas.after(self.refreshrate, self.__step)
        self.canvas.background('black')

    def __init__(self, width, height, hz=30):
        self.canvas = Canvas(width, height)
        self.refreshrate = hz
        self.delta = 1./ self.refreshrate
        self.done = False
        self.entities=set()

    def addEntity(self, entity):
        """ add something to the main loop"""
        self.entities.add(entity)

    def run(self):
        self.setup()
        self.__step()
        self.canvas.run()

    #override me:
    def setup(self):
        pass
    def step(self):
        pass
    def userinput(self):
        action = self.canvas.user_input.do()

        if action & Action.quit:
            self.canvas.user_input.print("Good Bye!")
            self.quit()
            return

    def quit(self):
        self.canvas.stop()
        self.done = True


#helper:
class P:
    decay = None
    point = None
    velocity = None
    direction = None
    color = None
    tag = None

    def __repr__(self):
        return f"P<{self.point},{self.velocity},"\
            f"{self.direction},decay={self.decay},{self.color},tag={self.tag}>"

class Particle:
    """ draw number particles, in direction, for duration steps """
    decay = 10
    velocities=[10, 13]
    colors=['white smoke', 'yellow1', 'wheat1','yellow','orange','lightgrey', 'NavajoWhite']
    spread = 30 # random deviation in direction
    wobble=20 #random size changes
    counter=0

    def __str__(self):
        return f"Particle<{self.coords}>"

    def __make(self):
        pi = P()
        pi.point = Point(self.coords.x, self.coords.y)
        pi.decay = self.decay
        pi.tag= f"parti{self.counter:x}"
        self.counter += 1
        pi.velocity = \
            random.randrange(self.velocities[0], self.velocities[1])
        pi.color = random.choice(self.colors)
        pi.direction = rotate(self.direction, (360 - self.spread/2) + random.random() * self.spread)
        return pi

    def set_position(self, point):
        self.coords = point

    def __init__(self, coords, direction, size=10, number=10, duration=30):
        self.size = size
        self.coords = coords
        self.active = True
        self.duration = duration
        self.direction = direction
        self.particles = []
        self.garbage=[]

        for _ in range(number):
            pi = self.__make()
            self.particles.append(pi)

    def activate(self, duration):
        self.duration = duration

    def done(self):
        return self.duration < 1

    def step(self):
        self.duration -= 1
        # TODO move particles
        cur = []
        for pi in self.particles:
            if pi.decay <= 0:
                #new generation
                self.garbage.append(pi.tag)
                npi = self.__make()
            else:
                npi = pi
            npi.point = npi.point + (npi.direction * npi.velocity )
            npi.decay -= 1
            cur.append(npi)
        self.particles = cur

    def draw(self, canvas):
        for tag in self.garbage:
            canvas.delete(tag)
        self.garbage.clear()

        for pi in self.particles:
            r = random.random() * self.wobble + self.size 
            pi.tag = canvas.draw_oval(pi.tag,
                    (pi.point.x, pi.point.y),
                    (pi.point.x + r, pi.point.y + r),
                    pi.color, pi.decay
                    )



class MondLander(Game):
    # config:
    g = 1.625 # m/s**2
    gV = Vector(0, 1 * g, 0)
    steering = 2 #vectored thrust 
    fuel_consumption = 5
    thrust = Vector(0, -5, 0)
    fatal_velocity = Vector(0, 1, 0)
    scale = 5

    def setup(self):
        self.lander = Lander(250, 0)
        self.lander.mesh = scale(self.lander.mesh, self.scale)
        self.lander.mesh = translate(self.lander.mesh, self.lander.position())
        self.particle = Particle(Point(250,0),Vector(0,-1,0), duration=0)

    def userinput(self):
        super().userinput()

        action = self.canvas.user_input.do()

        if not self.done:
            obj = self.lander.mesh

            if action & Action.left:
                obj = translate(obj, Vector(-1 * self.steering, 0, 0))
                self.lander.fuel -= self.fuel_consumption * self.delta
            if action & Action.right:
                obj = translate(obj, Vector(1 * self.steering, 0, 0))
                self.lander.fuel -= self.fuel_consumption * self.delta
            if action & Action.up:
                self.lander.velocity += self.thrust * self.delta
                self.lander.fuel -= self.fuel_consumption * self.delta
                self.particle.activate(self.refreshrate)

            self.lander.mesh = obj

    def step(self):
        """ simulate """
        obj = self.lander.mesh
        c = self.canvas
        #gravitation!1
        self.lander.velocity += self.gV * self.delta
        obj = translate(obj, self.lander.velocity )
        self.lander.mesh = obj
       
        # draw rocket exhaust
        if not self.particle.done():
            #TODO should be an entity
            # render particles as spheres
            self.particle.set_position(self.lander.position())
            self.particle.step() 
            self.particle.draw(c)

        # TODO draw moon surface, landing zone

        # TODO use grid for HUD
        c.draw_text("ALT", Point(c.width()-128, 10), "orange",
                f"ALT {altitude(obj,c):.1f}")
        c.draw_text("FUEL", Point(c.width()-128, 40), "orange",
                f"FUEL {self.lander.fuel:.1f}")
        c.draw_text("m/s", Point(c.width()-128, 70), "orange",
                f"m/s {abs(self.lander.velocity.y):.1f}")

        c.draw_polygon("Lander", obj, self.lander.color)

        # TODO do collision detection canvas borders
        h = altitude(obj, c)
        if h < 1:
            def banner(text):
                c.user_input.print(text)
                c.draw_text("GAMEOVER", Point(c.width()/2-len(text)*2, c.height()/2),
                        "orange", text)
            if self.lander.velocity.y  > self.fatal_velocity.y:
                banner(f"YOU CRASHED!")
            else:
                banner(f"YOU LANDED!")
            self.done = True

class TestParticle(Game):
    def setup(self):
        x,y   = self.canvas.width()/2, self.canvas.height()/2
        self.particle=Particle(Point(x,y), Vector(0,-1,0), 10, 1,
                duration=5 * self.refreshrate)
    def step(self):
        if self.particle.done():
            self.quit()
            return
        self.particle.step()
        self.particle.draw(self.canvas)

import argparse as Ap
import sys

if __name__ == "__main__":
    ap = Ap.ArgumentParser("Mondlander")
    ap.add_argument("--test",  help="Run unit tests", action="store_true")
    ap.add_argument("--test-anim", help="Run animation tests", action="store_true")
    ap.add_argument("--test-particle", help="Run particle tests", action="store_true")
    args = ap.parse_args()

    if args.test:
        unittest.main(argv=[sys.argv[0], "--verbose"])
    elif args.test_anim:
        animTest()
    elif args.test_particle:
        TestParticle(400, 300).run()
    else:
        MondLander(640, 480).run()
