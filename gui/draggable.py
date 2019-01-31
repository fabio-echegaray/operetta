import matplotlib.patches
from sympy import asinh, lambdify, log, sqrt, symbols
from sympy.physics.mechanics import ReferenceFrame
import shapely.geometry
import numpy as np
from sympy.solvers.solvers import nsolve
import seaborn as sns
from descartes import PolygonPatch


class DraggableCircle:
    def __init__(self, circle, callback=None):
        if type(circle) != matplotlib.patches.Circle: raise Exception('not a circle')
        self.circle = circle
        self.press = None
        self.callfn = callback

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.circle.figure.canvas.mpl_connect('button_press_event', lambda s: self.on_press(s))
        self.cidrelease = self.circle.figure.canvas.mpl_connect('button_release_event', lambda s: self.on_release(s))
        self.cidmotion = self.circle.figure.canvas.mpl_connect('motion_notify_event', lambda s: self.on_motion(s))

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

        if self.callfn is not None:
            self.callfn()


class DraggableEllipse:
    def __init__(self, ellipse, callback=None):
        if type(ellipse) != matplotlib.patches.Ellipse: raise Exception('not a ellipse')
        self.ellipse = ellipse
        self.press = None
        self.callfn = callback

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.ellipse.figure.canvas.mpl_connect('button_press_event', lambda s: self.on_press(s))
        self.cidrelease = self.ellipse.figure.canvas.mpl_connect('button_release_event', lambda s: self.on_release(s))
        self.cidmotion = self.ellipse.figure.canvas.mpl_connect('motion_notify_event', lambda s: self.on_motion(s))

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
        print('event contains', self.ellipse.center)
        x0, y0 = self.ellipse.center
        self.press = x0, y0, self.ellipse.width, self.ellipse.height, self.ellipse.angle, event.xdata, event.ydata

    def on_motion(self, event):
        'on motion we will move the circle if the mouse is over us'
        if self.press is None: return
        if event.inaxes != self.ellipse.axes: return
        x0, y0, w, h, a, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.ellipse.center = (x0 + dx, y0 + dy)
        self.ellipse.figure.canvas.draw()

        # print('x0=%f, xpress=%f, event.xdata=%f, dx=%f, x0+dx=%f' %
        #       (x0, xpress, event.xdata, dx, x0 + dx))

    def on_release(self, event):
        'on release we reset the press data'
        self.press = None
        self.ellipse.figure.canvas.draw()

        if self.callfn is not None:
            self.callfn()


class DraggableEightNote:
    def __init__(self, ax, ellipseG1, ellipseG2, circleHeight, number_of_sphase_segments=2):
        if type(ellipseG1) != matplotlib.patches.Ellipse or type(ellipseG2) != matplotlib.patches.Ellipse:
            raise Exception('inputs are not an ellipse')
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider

        self.e1 = ellipseG1
        self.e2 = ellipseG2
        self.c = circleHeight
        self._ax = ax
        self._polygons = None
        self._n_sphase = number_of_sphase_segments

        ax.add_artist(ellipseG1)
        ax.add_artist(ellipseG2)
        ax.add_artist(circleHeight)
        dp1 = DraggableEllipse(ellipseG1)
        dp2 = DraggableEllipse(ellipseG2)
        dc = DraggableCircle(circleHeight)

        dc.callfn = self.update
        dp2.connect()
        dp1.connect()
        dc.connect()

        axcolor = 'lightgoldenrodyellow'
        axe1 = plt.axes([0.25, 0.35, 0.65, 0.03], facecolor=axcolor)  # [left, bottom, width, height]
        axe2 = plt.axes([0.25, 0.3, 0.65, 0.03], facecolor=axcolor)
        axe3 = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)
        axe4 = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)
        axe5 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
        axe6 = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        we1, he1, ae1 = self.e1.width, self.e1.height, self.e1.angle
        we2, he2, ae2 = self.e2.width, self.e2.height, self.e2.angle
        self.we1 = Slider(axe1, 'Width G1', 0.1, 5.0, valinit=we1)
        self.he1 = Slider(axe2, 'Height G1', 0.1, 5.0, valinit=he1)
        self.ae1 = Slider(axe3, 'Angle G1', 0, 180, valinit=ae1)
        self.we2 = Slider(axe4, 'Width G2', 0.1, 5.0, valinit=we2)
        self.he2 = Slider(axe5, 'Height G2', 0.1, 5.0, valinit=he1)
        self.ae2 = Slider(axe6, 'Angle G2', 0, 180, valinit=ae1)
        # self.sw = Slider(axe6, 'S-phase width', 0, 3, valinit=we2/2)

        for slider in [self.we1, self.he1, self.ae1, self.we2, self.he2, self.ae2]:
            slider.on_changed(self.update_slider)

        self.update()

    def update_slider(self, val):
        self.e1.width = self.we1.val
        self.e2.width = self.we2.val
        self.e1.height = self.he1.val
        self.e2.height = self.he2.val
        self.e1.angle = self.ae1.val
        self.e2.angle = self.ae2.val
        self._ax.figure.canvas.draw()

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

    def _calc_polygons(self):
        xe1, ye1 = self.e1.center
        we1, he1, ae1 = self.e1.width, self.e1.height, self.e1.angle
        xe2, ye2 = self.e2.center
        we2, he2, ae2 = self.e2.width, self.e2.height, self.e2.angle
        xc, yc = self.c.center
        self._polygons = None

        # -------------------
        # G1 ellipse
        # -------------------
        circ = shapely.geometry.Point((xe1, ye1)).buffer(0.5)
        ell = shapely.affinity.scale(circ, we1, he1)
        ellipseG1 = shapely.affinity.rotate(ell, ae1)
        self._polygons = [ellipseG1]

        # -------------------
        # G2 ellipse
        # -------------------
        circ = shapely.geometry.Point((xe2, ye2)).buffer(0.5)
        ell = shapely.affinity.scale(circ, we2, he2)
        ellipseG2 = shapely.affinity.rotate(ell, ae2)

        # -------------------
        # S-phase polygons
        # -------------------
        t = symbols('t', real=True, positive=True)
        ref = ReferenceFrame('N')
        funcs, lambdas, evals = self.s_phase_function(ref, x0=xe1, y0=ye1,
                                                      xf=xe2 + we2 / 2 * np.cos(np.radians(ae2)), yf=yc)
        f, tn, nn, s = funcs
        lamda_fx, lamda_fy, lambda_s = lambdas
        ta, fx_ev, fy_ev, s_ev = evals
        self.ta, self.fx_ev, self.fy_ev, self.s_ev = ta, fx_ev, fy_ev, s_ev

        # use normal vector to the curve to construct inner/outer paths
        h = we1 / 3
        interior = lambdify(t, (nn * h).to_matrix(ref))
        exterior = lambdify(t, (nn * -h).to_matrix(ref))

        # make a partition of the arclength based on the number of segments
        fn = f.dot(ref.y) - ye1
        root = float(nsolve(fn, (1e-20, 1), solver='bisect', tol=1e-30, verify=False, verbose=False))
        s_ini = lambda_s(root)
        s_partition = np.linspace(s_ini, s_ev[-1], num=self._n_sphase)

        # construct each polygon based on the s partition
        sti = None
        for k, st in enumerate(s_partition):
            s_ii = np.where(s_ev <= st)[0].max()
            _ta = ta[s_ii + 1 if s_ii + 1 < ta.size else s_ii]

            xi, yi, _ = interior(_ta) + np.array([[lamda_fx(_ta)], [lamda_fy(_ta)], [0]])
            xf, yf, _ = exterior(_ta) + np.array([[lamda_fx(_ta)], [lamda_fy(_ta)], [0]])

            if sti is not None:
                s_ix = np.where((sti <= s_ev) & (s_ev <= st))[0]
                _ta = ta[np.append(s_ix, s_ix.max() + 1 if s_ix.max() + 1 < ta.size else s_ix.max())]
                [xm], [ym], _ = np.array([[lamda_fx(_ta)], [lamda_fy(_ta)], [0]])
                [xi], [yi], _ = interior(_ta) + np.array([[xm], [ym], [0]])
                [xf], [yf], _ = exterior(_ta) + np.array([[xm], [ym], [0]])
                pointList = list()
                pointList.extend([shapely.geometry.Point(x, y) for x, y in zip(xi, yi)])
                pointList.extend([shapely.geometry.Point(x, y) for x, y in zip(np.flip(xf), np.flip(yf))])

                poly = shapely.geometry.Polygon([(p.x, p.y) for p in pointList])
                self._polygons.append(poly - ellipseG1)
            sti = st

        # construct last polygon of S-phase
        self._last_x = np.mean(xm)
        self._last_y = np.mean(ym)
        pointList = list()
        pointList.extend([shapely.geometry.Point(x, y) for x, y in zip(np.flip(xi), np.flip(yi))])
        # re2 = np.radians(ae2)
        # m = np.tan(re2)
        # dx = (lambda_s(_ta[-1])- lambda_s(_ta[0]))/ 2
        # dy = m * dx
        # pointList.append(shapely.geometry.Point(xe2 - dx, ye2 - dy))
        # pointList.append(shapely.geometry.Point(xe2 + dx, ye2 + dy))
        # np.sqrt((yi - ye2) ** 2)
        pointList.extend([shapely.geometry.Point(x, y) for x, y in zip(xi, yi - np.abs(np.mean(yi) - ye2))])

        poly = shapely.geometry.Polygon([(p.x, p.y) for p in pointList])
        self._polygons.append(poly - ellipseG2)

        # yield final ellipse
        self._polygons.append(ellipseG2)

    def polygons(self):
        self._calc_polygons()
        for p in self._polygons:
            yield p

    def clear(self):
        # self._ax.artists = []
        # self._ax.collections = []
        self._ax.lines = []
        self._ax.texts = []
        self._ax.patches = []

    def update(self):
        self.clear()
        self._calc_polygons()

        xe2, ye2 = self.e2.center
        self._ax.plot(self.fx_ev, self.fy_ev)
        # self._ax.scatter(self.fx_ev, self.fy_ev, marker='+')
        self._ax.plot([self._last_x, xe2], [self._last_y, ye2])

        current_palette = sns.color_palette('bright', n_colors=10)
        for i, p in enumerate(self._polygons):
            patch = PolygonPatch(p, fc=current_palette[i], ec="#999999", alpha=0.5, zorder=2)
            self._ax.text(p.centroid.x, p.centroid.y, i, fontsize=12, color='white')
            self._ax.add_patch(patch)

        self._ax.figure.canvas.draw()
