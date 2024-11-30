# Installation

### Requirements

所有代码都在以下环境中进行测试：

* Ubuntu 22.04 ROS2 Humble
* Python 3.10
* `numpy<2`

### Docker

推荐使用，假定已经安装好 docker

```bash
echo '[[ -x "$(command -v xhost)" ]] && xhost +' >> ~/.bashrc
docker pull city945/rospyutils:humble-desktop-full
# 将 -v 参数挂载的目录替换为你的工作区路径，在图形界面终端中运行而非 SSH 连接的终端
docker run -itd --ipc=host --network=host --privileged=true -e DISPLAY=unix$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /datasets:/datasets -v /home/city945/workspace:/workspace --name rospyutils city945/rospyutils:humble-desktop-full zsh
# 下载 VSCode 插件 Docker 和 Dev Containers, Attach Visual Studio Code 进入容器内开发
```

！！！注意：容器内家目录下有一些个人配置文件，如出现非预期行为请删除相关的配置文件后使用

### Shell Script

假定已经安装好 ros2

```bash
apt update ; apt install -y python3-pip gdb
pip3 install 'numpy<2' opencv-python 'pu4c>=1.3.1' pandas easydict # CvBridge needs numpy<2
# pip3 install nuscenes-devkit==1.1.11 # 可选
```
