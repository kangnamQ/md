

# [우분투 , 모니터 없는 서버 vnc 접속시 해상도 조정]

참조 : (https://blog.1day1.org/610)

---

```
apt install xserver-xorg-video-dummy
```

xorg 설정을 해준다. ( /usr/share/X11/xorg.conf.d/xorg.conf )

```
Section "Device"
    Identifier "DummyDevice"
    Driver "dummy"
    VideoRam 256000
EndSection

Section "Screen"
    Identifier "DummyScreen"
    Device "DummyDevice"
    Monitor "DummyMonitor"
    DefaultDepth 24
    SubSection "Display"
        Depth 24
        Modes "1920x1080_60.0"
    EndSubSection
EndSection

Section "Monitor"
    Identifier "DummyMonitor"
    HorizSync 30-70
    VertRefresh 50-75
    ModeLine "1920x1080" 148.50 1920 2448 2492 2640 1080 1084 1089 1125 +Hsync +Vsync
EndSection
```

재부팅하면 원하는 해상도로 접속이 된다.





// but 모니터 끼면 모니터에는 안보임 -_-...



혹시나 문제 생겼을 경우 비상부팅 방법

```
GNU GRUB 화면에서

Ubuntu 에 커서를 두고 'e'
끝쪽에 'linux ~~~~ $vt_handoff' 라고 되어있는 부분을 지우고

'systemd.unit=emergency.target' 을적고 'ctrl-x'누르면 비상부팅

엔터 

혹은 'cat /etc/fstab' 하면 info줌

"시스템을 변경하려면 그림과 같이 읽기 및 쓰기 모드로 마운트해야합니다."

# mount -o remount,rw /

"여기에서 표시된대로 루트 암호 변경과 같은 문제 해결 작업을 수행 할 수 있습니다. ""완료되면 변경 사항이 적용되도록 재부팅하십시오."

# systemctl reboot
```





210621//만약 부트로더 꼬였을경우

ubuntu live usb 만들어서

거기서 해당경로로 들어가서 해결하면됨

파일을 없에면 알아서 돌아옴.