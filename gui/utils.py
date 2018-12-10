from PIL import Image

def get_crop_bbox(image, qlabel, r, c):
    maxw, maxh = image.shape
    minr = int(r - qlabel.height() / 2.0)
    maxr = int(r + qlabel.height() / 2.0)
    minc = int(c - qlabel.width() / 2.0)
    maxc = int(c + qlabel.width() / 2.0)
    minr = minr if minr > 0 else 0
    minc = minc if minc > 0 else 0
    maxr = maxr if maxr < maxw else maxw
    maxc = maxc if maxc < maxh else maxh
    if maxr - minr < qlabel.height():
        if minr == 0:
            maxr = qlabel.height()
        else:
            minr = maxr - qlabel.height()
    if maxc - minc < qlabel.width():
        if minc == 0:
            maxc = qlabel.width()
        else:
            minc = maxc - qlabel.width()

    return minr, maxr, minc, maxc


def canvas_to_pil(canvas):
    canvas.draw()
    s = canvas.tostring_rgb()
    w, h = canvas.get_width_height()[::-1]
    im = Image.frombytes("RGB", (w, h), s)
    return im
