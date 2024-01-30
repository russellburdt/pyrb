
"""
short flask utility methods
R Burdt, 1 Jan 2018
"""


import flask

class Flask(flask.Flask):
    """
    wrapper around flask.Flask that prevents caching of .js files
    """

    def get_send_file_max_age(self, name):
        if name.lower().endswith('.js'):
            return 0
        return flask.Flask.get_send_file_max_age(self, name)

