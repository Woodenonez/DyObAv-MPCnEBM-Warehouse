## ROS2
ros2 run gazebo_map_creator request_map.py -c '(-14.0,-36.0,0.03)(14.0,23.0,0.5)' -r 0.1 -f $PWD/map --skip-vertical-scan

T1: gazebo -s libgazebo_map_creator.so ../DyObAv-MPCnEBM-Warehouse-ROS2/src/gazebo_worlds/worlds/aws/arena_hospital_small.world
T2: ros2 run gazebo_map_creator request_map.py -c '(-18.0,-2.0,0.03)(18.0,8.0,0.5)' -r 0.1 -f $PWD/map --skip-vertical-scan

## ROS1

>> sudo apt-get install libgazebo11-dev

>> rosdep install --from-paths src --ignore-src -r -y

>> git clone https://github.com/Hadi-Hajieghrary/mir_robot.git

>> git clone https://github.com/locusrobotics/robot_navigation.git

>> git clone https://github.com/uos/rospy_message_converter.git

>> git clone https://github.com/andreasBihlmaier/pysdf.git

>> git clone https://github.com/marinaKollmitz/gazebo_ros_2Dmap_plugin.git

>> cd pysdf

>> sudo python3 setup.py install

>> cd ros_ws

>> rosdep install --from-paths src --ignore-src -r -y

>> catkin build -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.8m

Add gazebo_occupancy_map plugin to the gazebo world.

<plugin name='gazebo_occupancy_map' filename='libgazebo_2Dmap_plugin.so'>
    <map_resolution>0.10</map_resolution> <!-- in meters, optional, default 0.1 -->
    <map_z>0.72</map_z>         <!-- in meters, optional, default 0.3 -->
    <map_size_x>20</map_size_x>          <!-- in meters, optional, default 10 -->
    <map_size_y>20</map_size_y>          <!-- in meters, optional, default 10 -->
    <init_robot_x>0</init_robot_x>          <!-- x coordinate in meters, optional, default 0 -->
    <init_robot_y>0</init_robot_y>          <!-- y coordinate in meters, optional, default 0 -->
</plugin>

Use the following ros service call to generate the map.

>> rosservice call /gazebo_2Dmap_plugin/generate_map

And, save the map:

>> rosrun map_server map_saver -f <mapname> /map:=/map2d




