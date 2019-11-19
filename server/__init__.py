from .app import App
import os

# - Get the project path
path = os.getcwd()

# - Server application
app = App(path)
