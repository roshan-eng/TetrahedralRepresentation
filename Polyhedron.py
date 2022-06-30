import sys
import math
import ctypes
import numpy as np
import OpenGL.GL as gl
import OpenGL.GLUT as glut
from itertools import combinations


# Class Polyhedron deals with opengl modelling
class Polyhedron:

    def __init__(self):
        self.vertex_code = """
        uniform mat4   u_model;         // Model matrix
        uniform mat4   u_view;          // View matrix
        uniform mat4   u_projection;    // Projection matrix
        uniform vec4   u_color;         // Global color

        attribute vec4 a_color;         // Vertex color
        attribute vec3 a_position;      // Vertex position

        varying vec4   v_color;         // Interpolated fragment color (out)
        void main()
        {
            v_color = u_color * a_color;
            gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
        }
        """

        self.fragment_code = """
        varying vec4 v_color;         // Interpolated fragment color (in)
        void main()
        {
            gl_FragColor = v_color;
        }
        """

        self.gpu = None
        self.phi_ = 0
        self.theta_ = 0

        # Build Polyhedron
        # Provide all the necessary inputs to build a polyhedron (Tetrahedron in our case)
        # --------------------------------------

        self.vertices = np.zeros(8, [("a_position", np.float32, 3),
                                     ("a_color", np.float32, 4)])
        self.vertices["a_position"] = [[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1],
                                       [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1]]
        self.vertices["a_color"] = [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 1, 0, 1],
                                    [1, 1, 0, 1], [1, 1, 1, 1], [1, 0, 1, 1], [1, 0, 0, 1]]
        self.f_indices = np.array([0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5, 0, 5, 6, 0, 6, 1,
                                   1, 6, 7, 1, 7, 2, 7, 4, 3, 7, 3, 2, 4, 7, 6, 4, 6, 5], dtype=np.uint32)
        self.o_indices = np.array([0, 1, 1, 2, 2, 3, 3, 0, 4, 7, 7, 6,
                                   6, 5, 5, 4, 0, 5, 1, 6, 2, 7, 3, 4], dtype=np.uint32)

    @staticmethod
    def rotate(M, angle, x, y, z):
        angle = math.pi * angle / 180
        c, s = math.cos(angle), math.sin(angle)
        n = math.sqrt(x * x + y * y + z * z)
        x, y, z = x / n, y / n, z / n
        cx, cy, cz = (1 - c) * x, (1 - c) * y, (1 - c) * z
        R = np.array([[cx * x + c, cy * x - z * s, cz * x + y * s, 0],
                      [cx * y + z * s, cy * y + c, cz * y - x * s, 0],
                      [cx * z - y * s, cy * z + x * s, cz * z + c, 0],
                      [0, 0, 0, 1]], dtype=M.dtype).T
        M[...] = np.dot(M, R)
        return M

    @staticmethod
    def translate(M, x, y=None, z=None):
        y = x if y is None else y
        z = x if z is None else z
        T = np.array([[1.0, 0.0, 0.0, x],
                      [0.0, 1.0, 0.0, y],
                      [0.0, 0.0, 1.0, z],
                      [0.0, 0.0, 0.0, 1.0]], dtype=M.dtype).T
        M[...] = np.dot(M, T)
        return M

    @staticmethod
    def frustum(left, right, bottom, top, z_near, z_far):
        M = np.zeros((4, 4), dtype=np.float32)
        M[0, 0] = +2.0 * z_near / (right - left)
        M[2, 0] = (right + left) / (right - left)
        M[1, 1] = +2.0 * z_near / (top - bottom)
        M[3, 1] = (top + bottom) / (top - bottom)
        M[2, 2] = -(z_far + z_near) / (z_far - z_near)
        M[3, 2] = -2.0 * z_near * z_far / (z_far - z_near)
        M[2, 3] = -1.0
        return M

    def perspective(self, f_ovy, aspect, z_near, z_far):
        h = math.tan(f_ovy / 360.0 * math.pi) * z_near
        w = h * aspect
        return self.frustum(-w, w, -h, h, z_near, z_far)

    def display(self):
        gl.glDepthMask(gl.GL_TRUE)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # Create Polyhedron using triangle mesh
        gl.glDisable(gl.GL_BLEND)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glUniform4f(self.gpu["uniform"]["u_color"], 1, 1, 1, 1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.gpu["buffer"]["filled"])
        gl.glDrawElements(gl.GL_TRIANGLES, len(self.f_indices), gl.GL_UNSIGNED_INT, None)

        # Outlined Model
        gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glEnable(gl.GL_BLEND)
        gl.glDepthMask(gl.GL_FALSE)
        gl.glUniform4f(self.gpu["uniform"]["u_color"], 0, 0, 0, .5)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.gpu["buffer"]["outline"])
        gl.glDrawElements(gl.GL_LINES, len(self.o_indices), gl.GL_UNSIGNED_INT, None)
        gl.glDepthMask(gl.GL_TRUE)

        # Rotate Model
        self.theta_ += 0.5  # degrees
        self.phi_ += 0.5  # degrees

        model = np.eye(4, dtype=np.float32)
        self.rotate(model, self.theta_, 0, 0, 1)
        self.rotate(model, self.phi_, 0, 1, 0)
        gl.glUniformMatrix4fv(self.gpu["uniform"]["u_model"], 1, False, model)
        glut.glutSwapBuffers()

    def reshape(self, width, height):
        gl.glViewport(0, 0, width, height)
        projection = self.perspective(45.0, width / float(height), 2.0, 100.0)
        gl.glUniformMatrix4fv(self.gpu["uniform"]["u_projection"], 1, False, projection)

    @staticmethod
    def keyboard(key, x, y):
        if key == b'\033':
            sys.exit()

    def timer(self, fps):
        glut.glutTimerFunc(1000 // fps, self.timer, fps)
        glut.glutPostRedisplay()

    def main(self):
        # GLUT init
        # --------------------------------------
        glut.glutInit()
        glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA | glut.GLUT_DEPTH)
        glut.glutCreateWindow('Woo,  Jhakaas  !!!')
        glut.glutReshapeWindow(2048, 1024)
        glut.glutReshapeFunc(self.reshape)
        glut.glutDisplayFunc(self.display)
        glut.glutKeyboardFunc(self.keyboard)
        glut.glutTimerFunc(1000 // 60, self.timer, 60)

        # Build & activate program
        # --------------------------------------
        program = gl.glCreateProgram()
        vertex = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        fragment = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        gl.glShaderSource(vertex, self.vertex_code)
        gl.glCompileShader(vertex)
        gl.glAttachShader(program, vertex)
        gl.glShaderSource(fragment, self.fragment_code)
        gl.glCompileShader(fragment)
        gl.glAttachShader(program, fragment)
        gl.glLinkProgram(program)
        gl.glDetachShader(program, vertex)
        gl.glDetachShader(program, fragment)
        gl.glUseProgram(program)

        # Build GPU objects
        # --------------------------------------
        self.gpu = {"buffer": {}, "uniform": {}}

        self.gpu["buffer"]["vertices"] = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.gpu["buffer"]["vertices"])
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, gl.GL_DYNAMIC_DRAW)
        stride = self.vertices.strides[0]

        offset = ctypes.c_void_p(0)
        loc = gl.glGetAttribLocation(program, "a_position")
        gl.glEnableVertexAttribArray(loc)
        gl.glVertexAttribPointer(loc, 3, gl.GL_FLOAT, False, stride, offset)

        offset = ctypes.c_void_p(self.vertices.dtype["a_position"].itemsize)
        loc = gl.glGetAttribLocation(program, "a_color")
        gl.glEnableVertexAttribArray(loc)
        gl.glVertexAttribPointer(loc, 4, gl.GL_FLOAT, False, stride, offset)

        self.gpu["buffer"]["filled"] = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.gpu["buffer"]["filled"])
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, self.f_indices.nbytes, self.f_indices, gl.GL_STATIC_DRAW)

        self.gpu["buffer"]["outline"] = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.gpu["buffer"]["outline"])
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, self.o_indices.nbytes, self.o_indices, gl.GL_STATIC_DRAW)

        # Bind uniforms
        # --------------------------------------
        self.gpu["uniform"]["u_model"] = gl.glGetUniformLocation(program, "u_model")
        gl.glUniformMatrix4fv(self.gpu["uniform"]["u_model"], 1, False, np.eye(4))

        self.gpu["uniform"]["u_view"] = gl.glGetUniformLocation(program, "u_view")
        view = self.translate(np.eye(4), 0, 0, -5)
        gl.glUniformMatrix4fv(self.gpu["uniform"]["u_view"], 1, False, view)

        self.gpu["uniform"]["u_projection"] = gl.glGetUniformLocation(program, "u_projection")
        gl.glUniformMatrix4fv(self.gpu["uniform"]["u_projection"], 1, False, np.eye(4))

        self.gpu["uniform"]["u_color"] = gl.glGetUniformLocation(program, "u_color")
        gl.glUniform4f(self.gpu["uniform"]["u_color"], 1, 1, 1, 1)

        self.phi_, self.theta_ = 0, 0

        # Enter mainloop
        # --------------------------------------
        gl.glClearColor(1, 1, 1, 1)
        gl.glPolygonOffset(1, 1)
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
        gl.glLineWidth(1.0)
        glut.glutMainLoop()

