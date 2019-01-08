import matplotlib.patches
from sympy import N, asinh, lambdify, log, sqrt, symbols
from sympy.physics.mechanics import ReferenceFrame
import shapely.geometry
import numpy as np
from sympy.solvers.solvers import nsolve


class DraggableCircle:
    def __init__(self, circle):
        if type(circle) != matplotlib.patches.Circle: raise Exception('not a circle')
        self.circle = circle
        self.press = None

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.circle.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.circle.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.circle.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.circle.figure.canvas.mpl_disconnect(self.cidpress)
        self.circle.figure.canvas.mpl_disconnect(self.cidrelease)
        self.circle.figure.canvas.mpl_disconnect(self.cidmotion)

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.circle.axes: return

        contains, attrd = self.circle.contains(event)
        if not contains: return
        print('event contains', self.circle.center)
        x0, y0 = self.circle.center
        self.press = x0, y0, self.circle.radius, event.xdata, event.ydata

    def on_motion(self, event):
        'on motion we will move the circle if the mouse is over us'
        if self.press is None: return
        if event.inaxes != self.circle.axes: return
        x0, y0, r, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.circle.center = (x0 + dx, y0 + dy)
        self.circle.figure.canvas.draw()

        # print('x0=%f, xpress=%f, event.xdata=%f, dx=%f, x0+dx=%f' %
        #       (x0, xpress, event.xdata, dx, x0 + dx))

    def on_release(self, event):
        'on release we reset the press data'
        self.press = None
        self.circle.figure.canvas.draw()


class DraggableEllipse:
    def __init__(self, ellipse):
        if type(ellipse) != matplotlib.patches.Ellipse: raise Exception('not a ellipse')
        self.ellipse = ellipse
        self.press = None

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.ellipse.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.ellipse.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.ellipse.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.ellipse.figure.canvas.mpl_disconnect(self.cidpress)
        self.ellipse.figure.canvas.mpl_disconnect(self.cidrelease)
        self.ellipse.figure.canvas.mpl_disconnect(self.cidmotion)

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.ellipse.axes: return

        contains, attrd = self.ellipse.contains(event)
        if not contains: return
        print('event contains', self.ellipse.get_center())
        x0, y0 = self.ellipse.get_center()
        self.press = x0, y0, self.ellipse.width, self.ellipse.height, self.ellipse.angle, event.xdata, event.ydata

    def on_motion(self, event):
        'on motion we will move the circle if the mouse is over us'
        if self.press is None: return
        if event.inaxes != self.ellipse.axes: return
        x0, y0, w, h, a, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.ellipse.set_center((x0 + dx, y0 + dy))
        self.ellipse.figure.canvas.draw()

        # print('x0=%f, xpress=%f, event.xdata=%f, dx=%f, x0+dx=%f' %
        #       (x0, xpress, event.xdata, dx, x0 + dx))

    def on_release(self, event):
        'on release we reset the press data'
        self.press = None
        self.ellipse.figure.canvas.draw()


class DraggableEightNote:
    def __init__(self, ellipseG1, ellipseG2, circleHeight):
        if type(ellipseG1) != DraggableEllipse or type(ellipseG2) != DraggableEllipse:
            raise Exception('not a draggable ellipse')

        self.e1 = ellipseG1.ellipse
        self.e2 = ellipseG2.ellipse
        self.c = circleHeight.circle

    @staticmethod
    def s_phase_function(ref, x0=0, y0=0, xf=0, yf=1):
        t = symbols('t', real=True, positive=True)
        _a, _b = symbols('a b', real=True, positive=True)

        fx = t + _b
        fy = log(t, 10) + _a
        f = fx * ref.x + fy * ref.y

        # arclength=integrate(sqrt(fx.diff(t)**2+fy.diff(t)**2),t)
        # pprint(N(arclength, 3))
        arclength = t ** 2 * log(10) / sqrt(t ** 2 * log(10) ** 2 + 1) - asinh(1 / (t * log(10))) / log(10) + 1 / (
                sqrt(t ** 2 * log(10) ** 2 + 1) * log(10))

        a0 = yf + np.log(10) / 10
        _dx = 10 ** (y0 - a0)
        b0 = x0 - _dx
        f = f.subs({_a: a0, _b: b0})
        arclength = arclength - arclength.subs(t, _dx)
        arclength = arclength.subs({_a: a0, _b: b0})

        # print(arclength)
        # print(f)

        Tn = f.diff(t, ref).normalize().simplify()
        Nn = Tn.diff(t, ref).normalize().simplify()

        # evaluation
        tf = xf - b0
        ta = np.linspace(_dx, tf, num=100)
        f_x = lambdify(t, f.to_matrix(ref)[0])
        f_y = lambdify(t, f.to_matrix(ref)[1])
        s = lambdify(t, arclength)

        return (f, Tn, Nn, arclength), (f_x, f_y, s), (ta, f_x(ta), f_y(ta), s(ta))

    def polygons(self, number_of_sphase_segments=4):
        xe1, ye1 = self.e1.get_center()
        we1, he1, ae1 = self.e1.width, self.e1.height, self.e1.angle
        xe2, ye2 = self.e2.get_center()
        we2, he2, ae2 = self.e2.width, self.e2.height, self.e2.angle
        xc, yc = self.c.center

        # G1 ellipse
        circ = shapely.geometry.Point((xe1, ye1)).buffer(0.5)
        ell = shapely.affinity.scale(circ, we1, he1)
        ellipseG1 = shapely.affinity.rotate(ell, ae1)
        yield ellipseG1

        # G2 ellipse
        circ = shapely.geometry.Point((xe2, ye2)).buffer(0.5)
        ell = shapely.affinity.scale(circ, we2, he2)
        ellipseG2 = shapely.affinity.rotate(ell, ae2)

        t = symbols('t', real=True, positive=True)
        ref = ReferenceFrame('N')
        funcs, lambdas, evals = self.s_phase_function(ref, x0=xe1, y0=ye1, xf=xe2, yf=yc)
        f, tn, nn, s = funcs
        lamda_fx, lamda_fy, lambda_s = lambdas
        ta, fx_ev, fy_ev, s_ev = evals

        h = we1 / 2
        interior = lambdify(t, (nn * h).to_matrix(ref))
        exterior = lambdify(t, (nn * -h).to_matrix(ref))

        fn = f.dot(ref.y) - ye1
        root = float(nsolve(fn, (1e-20, 1), solver='bisect', tol=1e-30, verify=False, verbose=False))
        s_ini = lambda_s(root)
        print(-he1 / 2 - ye1)
        print(fn)
        print(N(fn.subs(t, root)))
        print(root, lambda_s(root), lamda_fx(root), lamda_fy(root))
        # s_partition=np.arange(s_ini, s_ev[-1], step=(s_ev[-1]-s_ini)/number_of_sphase_segments)
        s_partition = np.linspace(s_ini, s_ev[-1], num=number_of_sphase_segments)
        sti = None
        for k, st in enumerate(s_partition):
            print(k, st)
            s_ii = np.where(s_ev <= st)[0].max()
            _ta = ta[s_ii + 1 if s_ii + 1 < ta.size else s_ii]

            xi, yi, _ = interior(_ta) + np.array([[lamda_fx(_ta)], [lamda_fy(_ta)], [0]])
            xf, yf, _ = exterior(_ta) + np.array([[lamda_fx(_ta)], [lamda_fy(_ta)], [0]])

            if sti is not None:
                s_ix = np.where((sti <= s_ev) & (s_ev <= st))[0]
                _ta = ta[np.append(s_ix, s_ix.max() + 1 if s_ix.max() + 1 < ta.size else s_ix.max())]
                [xi], [yi], _ = interior(_ta) + np.array([[lamda_fx(_ta)], [lamda_fy(_ta)], [0]])
                [xf], [yf], _ = exterior(_ta) + np.array([[lamda_fx(_ta)], [lamda_fy(_ta)], [0]])
                pointList = list()
                pointList.extend([shapely.geometry.Point(x, y) for x, y in zip(xi, yi)])
                pointList.extend([shapely.geometry.Point(x, y) for x, y in zip(np.flip(xf), np.flip(yf))])

                poly = shapely.geometry.Polygon([(p.x, p.y) for p in pointList])
                yield poly - ellipseG1

            sti = st

        # construct last polygon of S-phase
        pointList = list()
        pointList.extend([shapely.geometry.Point(x, y) for x, y in zip(np.flip(xi), np.flip(yi))])
        re2 = np.radians(ae2)
        m = np.tan(re2)
        dx = (we2 / 2 + np.cos(re2)) / 4
        dy = m * dx
        pointList.append(shapely.geometry.Point(xe2 - dx, ye2 - dy))
        pointList.append(shapely.geometry.Point(xe2 + dx, ye2 + dy))

        poly = shapely.geometry.Polygon([(p.x, p.y) for p in pointList])
        yield poly - ellipseG2

        # yield final ellipse
        yield ellipseG2
