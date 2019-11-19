import os, sys, yaml
################################################################################
################################################################################
################################################################################
###                                 SERVER                                   ###
################################################################################
################################################################################
################################################################################
from . import app

# - Get the project path
path = os.getcwd()

#   Run as developer mode
#
#   PARAMETERS:
#       -c : configuration file path
#
if __name__ == "__main__":

    # - Get parameters
    c_idx = sys.argv.index('-c') if '-c' in sys.argv else -1

    # - Get the configuration file
    config_path = sys.argv[c_idx + 1] if c_idx >= 0 else path + "/config.yaml"
    config_content = open(config_path).read()
    config = yaml.load(config_content, Loader=yaml.FullLoader)

    # - Set app configuration
    app.configure(config)

    # - Start webserver
    app.run()
