import math

# Definicion sala cuadrilatero

class Quadrilateral:
    """
    Quadrilateral class defines the main elements of a quadrilateral,
    being a, b, c, d the lengths of each of the four sides, and alpha,
    beta, gamma and delta being the angles of the corners.
    """

    def __init__(self, a, b, c, d, alpha, beta, gamma, delta):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta


class Room(Quadrilateral):
    def __init__(self, a, b, c, d, alpha, beta, gamma, delta, height):
        super().__init__(a, b, c, d, alpha, beta, gamma, delta)
        self.height = height

        self.vector = []
        self.set_vector()

    def set_vector(self):
        self.vector = [round(self.a), round(self.b), round(self.c), round(self.d),
                       round(self.alpha), round(self.beta), round(self.gamma), round(self.delta),
                       round(self.height)]

    def return_vector(self):
        return self.vector


class UTSRoom(Room):
    def __init__(self, a, b, c, d, alpha, beta, gamma, delta, height, grid_center, rt60):
        super().__init__(a, b, c, d, alpha, beta, gamma, delta, height)
        self.grid_center = grid_center
        self.rt60 = rt60

    def get_m_l_position(self, characteristics):
        zone = characteristics[1]
        array = characteristics[2]
        l = int(characteristics[3])
        m = int(characteristics[4])

        xl = round(-150 * math.sin((2 * l - 1) * math.pi / 60)) + self.grid_center[0]
        yl = round(150 * math.cos((2 * l - 1) * math.pi / 60)) + self.grid_center[1]
        zl = 145
        xm = 0
        ym = 0
        zm = 145

        if array == 'Planar':
            if zone == "A":
                xm = -14 + (4 * ((m - 1) % 8)) - 40 + self.grid_center[0]
                ym = 14 - (4 * math.floor(((m - 1) / 8))) + self.grid_center[1]
            elif zone == "B":
                xm = -14 + (4 * ((m - 1) % 8)) + 40 + self.grid_center[0]
                ym = 14 - (4 * math.floor(((m - 1) / 8))) + self.grid_center[1]
            elif zone == "C":
                xm = -14 + (4 * ((m - 1) % 8)) + self.grid_center[0]
                ym = 14 - (4 * math.floor(((m - 1) / 8))) + 40 + self.grid_center[1]
            elif zone == "D":
                xm = -14 + (4 * ((m - 1) % 8)) + self.grid_center[0]
                ym = 14 - (4 * math.floor(((m - 1) / 8))) - 40 + self.grid_center[1]
            elif zone == "E":
                xm = -14 + (4 * ((m - 1) % 8)) + self.grid_center[0]
                ym = 14 - (4 * math.floor(((m - 1) / 8))) + self.grid_center[1]

        elif array == 'Circular':
            rm = 12 - (2 * math.floor((m - 1) / 30))
            if zone == "A":
                xm = - rm * math.sin(((m - 1) % 30) * 2 * math.pi / 30) - 40 + self.grid_center[0]
                ym = rm * math.cos(((m - 1) % 30) * 2 * math.pi / 30) + self.grid_center[1]
            elif zone == "B":
                xm = - rm * math.sin(((m - 1) % 30) * 2 * math.pi / 30) + 40 + self.grid_center[0]
                ym = rm * math.cos(((m - 1) % 30) * 2 * math.pi / 30) + self.grid_center[1]
            elif zone == "C":
                xm = - rm * math.sin(((m - 1) % 30) * 2 * math.pi / 30) + self.grid_center[0]
                ym = rm * math.cos(((m - 1) % 30) * 2 * math.pi / 30) + 40 + self.grid_center[1]
            elif zone == "D":
                xm = - rm * math.sin(((m - 1) % 30) * 2 * math.pi / 30) + self.grid_center[0]
                ym = rm * math.cos(((m - 1) % 30) * 2 * math.pi / 30) - 40 + self.grid_center[1]
            elif zone == "E":
                xm = - rm * math.sin(((m - 1) % 30) * 2 * math.pi / 30) + self.grid_center[0]
                ym = rm * math.cos(((m - 1) % 30) * 2 * math.pi / 30) + self.grid_center[1]

        return [round(xl), round(yl), round(zl), round(xm), round(ym), round(zm), self.rt60]

    def return_embedding(self, characteristics):
        lis_mic_vector = self.get_m_l_position(characteristics)
        room_vector = self.return_vector()
        return room_vector + lis_mic_vector


def return_room(emb):
    name = None
    if emb[0] == 490:
        name = 'Anechoic'
    if emb[0] == 355:
        name = 'Small'
    if emb[0] == 736:
        name = 'Medium'
    if emb[0] == 994:
        name = 'Large'
    if emb[0] == 600:
        name = 'Box'


    return name


if __name__ == "__main__":

    Anechoic_Room = UTSRoom(490, 722, 490, 722, 90, 90, 90, 90, 529, [245, 361], 45)
    Hemi_Anechoic_Room = UTSRoom(490, 722, 490, 722, 90, 90, 90, 90, 529, [245, 361], 52)
    Small_Room = UTSRoom(355, 410, 401, 378, 96, 90, 85, 88, 300, [175.5, 205], 497)
    Medium_Room = UTSRoom(736, 520, 650, 434.5, 81, 92, 98, 89, 300, [368, 217.5], 659)
    Large_Room = UTSRoom(994, 923, 1087, 1022, 81.4, 105, 81.3, 92.3, 300, [497, 486.25], 1281)
    Box_Room = UTSRoom(600, 1175, 600, 1175, 90, 90, 90, 90, 300, [300, 881.25], 667)

    zones = ['A', 'B', 'C', 'D', 'E']
    arrays = ['Planar', 'Circular']

    vector = Large_Room.return_embedding(['LargeMeetingRoom', 'B', 'Circular', 22, 1])
    # Los datos que se introducen son = [Sala, Zona, Array, Speaker, Micrófono] según están definidos los archivos wav

    print(vector)
    print(vector[0:4])
    print(vector[4:8])
    print(vector[8:9])
    print(vector[9:12])
    print(vector[12:15])
    print(vector[15:16])
    print(len(vector))

