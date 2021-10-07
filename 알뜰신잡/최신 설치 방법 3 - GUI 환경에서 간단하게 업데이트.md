최신 설치 방법 3 - GUI 환경에서 간단하게 업데이트

참고 :  https://blog.daum.net/kssoft/151

---



배포 저장소 등록

$ sudo apt-add-repository -y ppa:cappelikan/ppa

 

$ sudo apt update

 

GUI 커널 설치 관리 패키지 설치 mainline

$ sudo apt install mainline

 

우분투 앱 검색에서 mainline 검색해서 실행하면 아래와 같은 창이 나옵니다.

원하는 커널 버젼을 선택해서 install 버튼을 누러 설치 합니다. 제거는 remove 하시면 됩니다.

설치시 관리자 권한으로 설치해야 되므로, 관리자 비밀번호를 물어보는데, 입력하시고...

설치가 완료되면 재부팅 하시면 적용 됩니다.

 

재부팅후

$ uname –r

로 버젼을 확인 합니다.