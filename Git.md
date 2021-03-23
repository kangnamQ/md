Git 
==



```python
// 모든 파일이 Staged 상태로 바뀐다.
$ git add *
// 파일들의 상태를 확인한다.
$ git status
On branch master
Changes to be committed:
(use "git reset HEAD <file>..." to unstage)
  renamed:    README.md -> README
  modified:   CONTRIBUTING.md

```

이때, git reset HEAD [file] 명령어를 통해 git add를 취소할 수 있다.

- 뒤에 파일명이 없으면 add한 파일 전체를 취소한다.
- CONTRIBUTING.md 파일을 Unstaged 상태로 변경해보자.



```python
// CONTRIBUTING.md 파일을 Unstage로 변경한다.
$ git reset HEAD CONTRIBUTING.md
Unstaged changes after reset:
M	CONTRIBUTING.md

    
// 파일들의 상태를 확인한다.
$ git status
On branch master
Changes to be committed:
(use "git reset HEAD <file>..." to unstage)
  renamed:    README.md -> README
Changes not staged for commit:
(use "git add <file>..." to update what will be committed)
(use "git checkout -- <file>..." to discard changes in working directory)
  modified:   CONTRIBUTING.md

   
```

---



git commit 취소하기
---

#### commit 취소하기

- 완료한 commit을 취소해야 할 때가 있다.
  1. 너무 일찍 commit한 경우
  2. 어떤 파일을 빼먹고 commit한 경우 이때, git reset HEAD^ 명령어를 통해 git commit을 취소할 수 있다.

```python
$ git log

// [방법 1] commit을 취소하고 해당 파일들은 staged 상태로 워킹 디렉터리에 보존
$ git reset --soft HEAD^
// [방법 2] commit을 취소하고 해당 파일들은 unstaged 상태로 워킹 디렉터리에 보존
$ git reset --mixed HEAD^ // 기본 옵션
$ git reset HEAD^ // 위와 동일
$ git reset HEAD~2 // 마지막 2개의 commit을 취소
// [방법 3] commit을 취소하고 해당 파일들은 unstaged 상태로 워킹 디렉터리에서 삭제
$ git reset --hard HEAD^

```



---

commit message 변경하기
---

commit message를 잘못 적은 경우, git commit –amend 명령어를 통해 git commit message를 변경할 수 있다.

```git commit --amend```



#### TIP git reset 명령은 아래의 옵션과 관련해서 주의하여 사용해야 한다.

- reset 옵션

  - soft : index 보존(a dd한 상태, staged 상태), 워킹 디렉터리의 파일 보존. 즉 모두 보존.
  - mixed : index 취소(add하기 전 상태, unstaged 상태), 워킹 디렉터리의 파일 보존 (기본 옵션)
  - hard : index 취소(add하기 전 상태, unstaged 상태), 워킹 디렉터리의 파일 삭제. 즉 모두 취소.

  

#### TIP 만약 워킹 디렉터리를 원격 저장소의 마지막 commit 상태로 되돌리고 싶으면, 아래의 명령어를 사용한다.

- 단, 이 명령을 사용하면 원격 저장소에 있는 마지막 commit 이후의 워킹 디렉터리와 add했던 파일들이 모두 사라지므로 주의해야 한다.

```
// 워킹 디렉터리를 원격 저장소의 마지막 commit 상태로 되돌린다.
$ git reset --hard HEAD
```

참조 : https://gmlwjd9405.github.io/2018/05/25/git-add-cancle.html



---

Window에서 git 사용하기
==

[참조링크](https://cofs.tistory.com/421)



1. https://desktop.github.com/ 에서 GitHub Desktop을  설치한다.
2. 다운받은 파일 `GitHubDesktopSetup.exe` 을 실행한다.
3. 설명해주는대로 진행한다.
4. 설치된 GitHub Desktop 바로가기에서 우클릭을 사용하여 파일위치 열기를 실행한다.
5. `\app-x.x.x\resources\app\git\cmd` 로 이동을 하면 git.exe파일이 있다.
6. 해당 파일의 송성정보에서 경로를 복사한다.
7. 내 pc(내컴퓨터) 우클릭 > 속성 > 고급 시스템 설정 > 고급 탭 > 환경변수 > 시스템 변수의 path 항목에 복사한 경로 추가
8. 등록후 cmd 창에서 git --version 명령어를 테스트해본다.