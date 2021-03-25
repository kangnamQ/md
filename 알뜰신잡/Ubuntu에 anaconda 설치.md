Ubuntu에 anaconda 설치하기 (가상환경)
==



참조 사이트
--

- [아나콘다 설치 참조 사이트](https://khann.tistory.com/21)
- [아나콘다 관련 명령어 참조 사이트](https://hamonikr.org/board_bFBk25/78585)
- [쥬피터 노트북 설치 참조 사이트](https://khann.tistory.com/22)

---



설치
---

[Anaconda 설치](https://www.anaconda.com/products/individual#Downloads,"anaconda 설치")

다운로드 들어가서 그냥 sh파일을 받아도 되고 다른방법을 사용하여 받아보겠다.



1. 다운로드 하려는 파일의 링크 복사 (오른쪽 클릭후 copy link address)

2. 설치하고자 하는 리눅스에 wget (복사한 링크 붙여넣기) 실행
   ` wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh`
3. 실행
   `bash Anaconda3-2020.11-Linux-x86_64.sh`
4. 엔터 후(하며?) 설치진행
   - 만약 경로를 바꾸고 싶으면 위치를 적고 엔터
   - by running conda init?
     - yes = .bashrc 파일에 환경변수 자동 기록
     - no = .bashrc 파일에 `export PATH=$HOME/anaconda3/bin:$PATH` 추가해주기
5. 쉘 시작시에 콘다의 기본환경을 비 활성화
   `conda config --set auto_activate_base false`

---



콘다 사용 방법
--

콘다 버전 확인 : `conda -V`

콘다 업데이트 : `conda -version`

아나콘다 가상환경 생성 : `conda create -n [생성할 가상환경이름] python=3`

생성된 가상환경 리스트 확인 : `conda env list`

가상환경 활성화 : `conda activate [가상환경]`

가상환경 비활성화 : `conda deactivate`

가상환경 삭제 :`conda env remove -n [가상환경]`

패키지 리스트 확인 : `conda list`

패키지 설치 : `conda install [패키지]`

패키지 제거 : `conda remove [패키지]`

패키지 업데이트 : `conda update [패키지]`

전체 패키지 업데이트 : `conda update -all`



~~[해당 변수] 의 내용을 쓰라는 것이지 괄호를 쓰면 안됩니다...~~



주피터 노트북 설치
--

1. notebook 설치
    `pip install notebook`

2. config 파일 생성
    `jupyter notebook --generate-config`

3. 서버비밀번호 생성

   ```
   $ python
   >>> from notebook.auth import passwd
   >>> passwd()
   Enter password : 생성할 비밀번호
   Verify password : 재확인 비밀번호
   '생성된 해쉬 정보'  <<<(이 생성된 해쉬 정보를 복사하기)
   
   ctrl + D (나가기)
   ```

4. 생성된 경로의 파일(~/.jupyter/jupyter_notebook_config.py) 실행
   `gedit ~/.jupyter/jupyter_notebook_config.py`

5. 설정하기

   여러가지 설정 가능한 기능들이 있으니 읽어보고 설정해도 되며 기본적인것은 아래와 같다.
   ctrl + F 를 활용해서 찾는것을 추천함

   - 비밀번호 설정 : c.NotebookApp.password 주석 해제 후 아까 생성된 해쉬 붙여넣기.
     (로컬에서 사용해서 다시 주석해놓음...)
   - 기본 작업 경로 설정 : c.NotebookApp.notebook_dir주석해제 후 실행하고자하는 dir 지정 (ex, /home/(사용자이름) )
   - 외부 접속 허용 : c.NotebookApp.allow_origin = '*'
   - 서버를 띄울 아이피 설정 : c.NotebookApp.ip = '0.0.0.0'
     (필자는 c.NotebookApp.ip = 'localhost'라고 설정함)
   - 주피터 서버 실행 시 브라우저 실행 X : c.NotebookApp.open_browser = False
     (나는 열고 싶으니 설정 안함)

6. 설정파일 저장후 닫기.

7. jupyter 실행
   `jupyter-notebook`

8. [service로 등록하여 부팅 시 자동으로 실행되게 하고 싶다면](https://khann.tistory.com/5)

---

