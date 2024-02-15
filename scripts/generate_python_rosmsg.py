import os
import shutil
from pathlib import Path
import subprocess

REPO_PATH = Path(__file__).absolute().parent.parent
PKG_PATH = REPO_PATH / "gsmini"
ROSMSG_DIR = REPO_PATH / "msg"
ros_ws_path = [parent for parent in PKG_PATH.parents if "_ws" in parent.name]
assert len(ros_ws_path) == 1
ros_ws_path = ros_ws_path[0]
TARGET_PYMSG_DIR = PKG_PATH / "msg"

print("ROSMSG_DIR:", ROSMSG_DIR)
print("ROS_WORKSPACE:", ros_ws_path)
print("TARGET_PYMSG_DIR:", TARGET_PYMSG_DIR)

# Scan the msg directory for message files
msg_files = [f for f in os.listdir(ROSMSG_DIR) if f.endswith(".msg")]

# Process each message file
for msg_file in msg_files:
    msg_name = msg_file.split(".")[0]

    # Find the compiled Python file for the message
    find_command = f"find {str(ros_ws_path)}/devel/lib -name _{msg_name}.py"
    msg_file = subprocess.check_output(find_command, shell=True, text=True)
    find_result = os.popen(find_command).read().strip()
    if find_result:
        print(find_result)
        # Copy the .py file to the python_package/msg folder
        shutil.copy(find_result, TARGET_PYMSG_DIR / f"_{msg_name}.py")
        # Update the __init__.py file to import the message
        with open(os.path.join(TARGET_PYMSG_DIR, "__init__.py"), "a") as init_file:
            init_file.write(f"from ._{msg_name} import *\n")

print("Process completed.")
