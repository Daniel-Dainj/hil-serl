source /home/dainanjun/codespaces/serl/catkin_ws/devel/setup.bash

export ROS_MASTER_URI=http://localhost:11311

uv run python serl_robot_infra/robot_servers/franka_server.py \
  --robot_ip=192.168.1.110 \
  --gripper_type=Franka \
  --flask_url=127.0.0.2 \
  --ros_port=11311
