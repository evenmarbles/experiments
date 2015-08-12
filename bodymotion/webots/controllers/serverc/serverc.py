import sys
import socket  # Import socket module
import numpy as np

from threading import Timer

# noinspection PyUnresolvedReferences
from controller import Supervisor
# noinspection PyUnresolvedReferences
from controller import Node
# noinspection PyUnresolvedReferences
from controller import Receiver


class Server(Supervisor):
    TIME_STEP = 64

    RECV_BUFFER = 256

    GOAL_X_LIMIT = 4.5
    GAOL_Z_LIMIT = 0.75

    GOAL_INVALID = -1
    GOAL_FAIL = 0
    GOAL_SUCCESS = 1

    def __init__(self, port):
        """
        Initialization of the supervisor.

        :param port: Port number reserved for service
        :type port: int
        """
        Supervisor.__init__(self)

        self._connections = []

        try:
            self._s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except socket.error, msg:
            print msg[1]

        try:
            host = socket.gethostname()
            self._s.bind((host, port))
        except socket.error, msg:
            print msg[1]

        self._s.listen(10)
        print "Waiting for a connection on port {0}...".format(port)

        # Set the server socket in non-blocking mode
        self._s.setblocking(0)

    def __del__(self):
        Supervisor.__del__(self)
        print "Cleanup"
        for conn in self._connections:
            conn.close()  # Close the connection
        self._s.close()

    def run(self):
        while True:
            # Perform a simulation step of 64 milliseconds
            # and leave the loop when the simulation is over
            if self.step(Server.TIME_STEP) == -1:
                break

            self._accept_connection()

            reset = False
            for conn in self._connections:
                try:
                    data = conn.recv(Server.RECV_BUFFER)
                    if not data:
                        continue
                    print "Received: ", data

                    if data == "request reset":
                        reset = True
                except socket.error:
                    pass

            if reset:
                for conn in self._connections:
                    conn.send("reset requested")
                    self.simulationRevert()

    def _accept_connection(self):
        try:
            conn, addr = self._s.accept()
            self._connections.append(conn)
            # noinspection PyStringFormat
            print 'Client (%s, %s) connected' % addr
        except socket.error:
            pass


def main(argv):
    """
    Main entry point
    """
    if len(argv) < 1:
        print usage()
        sys.exit(0)

    port = int(argv[0])

    controller = Server(port)
    controller.run()


def usage():
    return "Please specify the SUPERVISOR_PORT_NUMBER in the 'controllerArgs' field of the Supervisor robot."


if __name__ == "__main__":
    main(sys.argv[1:])
