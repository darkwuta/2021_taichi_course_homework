import taichi as ti

ti.init(arch=ti.gpu)

res = 500
scale = 5

pixels = ti.Vector.field(3, ti.uint8, shape=[res, res])
sandpile = ti.field(ti.i32, shape=[int(res / scale), int(res / scale)])
sandpile_buf = ti.field(ti.i32, shape=[int(res / scale), int(res / scale)])
iters = 1024


class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur


sand_pair = TexPair(sandpile, sandpile_buf)

@ti.kernel
def throw_sand():
    sand_pair.cur[int(res / scale) / 2 + 0.5, int(res / scale) / 2 + 0.5] += 1


@ti.kernel
def render():
    color = 0xE7E0AA
    for i, j in pixels:
        if sand_pair.cur[i / 5, j / 5] == 0:
            pixels[i, j] = [171, 182, 155]
        elif sand_pair.cur[i / 5, j / 5] == 1:
            pixels[i, j] = [234, 0, 118]
        elif sand_pair.cur[i / 5, j / 5] == 2:
            pixels[i, j] = [230, 178, 0]
        elif sand_pair.cur[i / 5, j / 5] == 3:
            pixels[i, j] = [0, 81, 28]


@ti.kernel
def evolve(sand_cur: ti.template(), sand_nxt: ti.template()):
    for i, j in sandpile:
        if sand_pair.cur[i, j] >= 4:
            sand_nxt[i + 1, j] += int(sand_cur[i, j] / 4)
            sand_nxt[i - 1, j] += int(sand_cur[i, j] / 4)
            sand_nxt[i, j - 1] += int(sand_cur[i, j] / 4)
            sand_nxt[i, j + 1] += int(sand_cur[i, j] / 4)
            sand_nxt[i, j] = 0


@ti.kernel
def copy_field():
    for I in ti.grouped(sandpile):
        sand_pair.nxt[I] = sand_pair.cur[I]


def main():
    gui = ti.GUI("SandPile Model", res)
    while gui.running:
        for j in range(iters):
            throw_sand()
            copy_field()
            evolve(sand_pair.cur, sand_pair.nxt)
            sand_pair.swap()
        render()
        gui.set_image(pixels)
        gui.show()


if __name__ == '__main__':
    main()
