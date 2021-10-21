

GIT
==



나의 git 주소 : https://github.com/kangnamQ

좀더 알아보고 싶다면 : https://backlog.com/git-tutorial/kr/

---

1. 주로쓰는 명령어
   --
```
1. git clone <URL>

2. git add .
	전체 add
2-1. git add <file> 
	특정파일만 add
2-2 git add *.py
	.py인 파일 모두 add

3. git commit -m "you want coment"

4. git push origin <branch name>

5. git config --list
5-1. git config [--global] user.name "name"
5-2. git config [--global] user.email "email" 

6. git status

7. git diff

8. git branch
8-1. git branch <name>

9. git checkout <name>
9-1. git checkout -b <name> 
	브런치를 만듬과 동시에 이동
	
10. git log

11-1. git reset --soft <commit number>
	작업환경은 변환 안되며 Head만 이동됨
11-2. git reset --hard <commit number>
	작업환경까지 변환됨
11-3. git reset --hard Head~1
	바로전 commit으로 이동 (commit 취소)

12. git init
	저장소 초기화

13. git remote -V
13-1. git remote add <저장소> <저장소 주소>

14. git pull
	원격 저장소에서 받아옴
(push가 github에 올리는거 / pull이 github에서 받아오는 것)

15. git rm <file_name>
	파일 삭제
15-1. git rm *.<파일형식>
15-2. git rm -rf <dir_name>
	디렉토리 삭제
15-3. git rm --cached <file_name>
	untracked 상태
	
16. git mv <file_name> <change_file_name>
	파일 이름 변경
16-1. git mv <file_name> <dst_dir>
16-2. git mv <file_name> <dst_dir/dst_file>
```

파일 용량이 크거나 이미 작업이 완료했던 백업파일을 올리거나 보관하고 싶다면

tags 기능을 사용하는게 훨신 좋음

tag기능 사용법은 아래쪽에 작성되어 있음

---



2. 알아두면 좋은 명령어
--

```
$ git init

$ git remote add origin <.git 주소>
	저장소 주소를 origin이란 이름으로 추가하겠다
$ git remote -v
	원격 저장소 목록 

	## 이미 사용하고 있는 폴더로 지정하고 싶다면 remote로 설정해주어도 괜찮지만
	$ git clone <저장소 주소> [로컬 저장소 디렉토리 이름]
	이후 .git 파일과 안에있는 파일을 사용하고 있는 폴더로 옮기면 자동으로 저장소가 지정되며 에러도 발생하지
	않아 더 편리한듯 하다.

$ git config user.name "<name>"
$ git config user.email "<email>"

$ git status

$ git add <file name, pattern, or directory>
$ git add .

# git commit -m '<commit message>'
$ git commit -m 'update readme.md'

# git push <원격 저장소> <브랜치>
$ git push origin master

# git clone <원격 저장소 주소> (<디렉토리명>)

# git pull <원격 저장소> <브랜치>
$ git pull origin master
```



3. git ignore
   --

   ```
   gedit .gitignore
   만들고 쓰면 됨
   
   
   ## 파일 무시 test.txt 
   ## 다음과 같은 확장자는 전체 무시 *.text *.exe *.zip 
   ## 폴더 무시 test/
   
   $ git add . 
   $ git commit -m "ignore file&folder config" 
   $ git push origin master
   
   ## 주의사항 ##
   기존의 git의 관리를 받고 있던(commit된 것들) 파일이나 폴더를 .gitignore 파일에 작성하고 
   add > commit > push 하여도 ignore(무시) 되지 않습니다.
   
   이럴때는 기존에 가지고 있는 cached를 치워야 합니다. 다음과 같은 명령어를 써주세요.
   
   ## 파일 이라면 git rm --cached test.txt 
   ## 전체파일 이라면 git rm --cached *.txt 
   ## 폴더 라면 git rm --cached test/ -r
   
   
   기존에 관리하고 있던(commit된 것들) 파일을 cached 명령어를 안쓰고 무시하는 방법이 있습니다.
   명령어는 다음과 같습니다.
   $ git update-index --assume-unchanged [파일명]
   
   다음과 같이 무시를 선언하고 다시 취소하는 방법도 있습니다.
   $ git update-index --no-assume-unchanged [파일명]
   
   수정사항 무시 파일 조회는 다음 명령어를 사용합니다.
   $ git ls-files -v|grep '^h'

   
   출처: https://kcmschool.com/194 [web sprit]
   ```
   
   

---

참고
--

## 4.1. 저장소를 처음 만들 때

1. GitHub에서 새로운 저장소를 만든다.

2. 로컬에서 디렉토리를 만들고 저장소를 초기화 한다.

   `git init`

3. 사용자 설정: 이 로컬 저장소를 수정하는 사람의 정보 등록 -> commit에 기록됨

   `git config [--global] user.name <사용자 이름>`

   `git config [--global] user.email <email 주소>`

4. 원격 저장소 지정하기

   `git remote add <저장소 별명 보통은 origin> <저장소 주소>`

5. 새 파일을 작성하거나 기존 파일을 저장소에 붙여놓고 파일을 **stage**하기 (버전 관리되는 파일로 등록하기)

   `git add <파일>` : 특정 파일만 stage하기

   `git add .` : 현재 폴더 아래 있는 모든 수정 사항 stage 하기

6. 언제나 현재 상태를 확인하고 싶을 땐

   `git status`

7. Stage한 저장소 전체의 상태를 복원 가능한 *버전*으로 메시지와 함께 저장하기

   `git commit -m <변경 사항 메시지>`

8. 파일 변경/추가 - add(5) - commit(7) 반복 하기

9. 원격 저장소에 그 동안 쌓은 commit들 업로드하기

   `git push origin master`

## 4.2. 새로운 환경에서 원격 저장소 받아서 작업하기

1. 원격 저장소를 복사한 로컬 저장소를 만든다. 디렉토리 이름을 지정하지 않으면 저장소 이름과 같은 이름의 디렉토리가 생긴다.

   `git clone <저장소 주소> [로컬 저장소 디렉토리 이름]`

   Git이 관리하는 디렉토리로 들어가야 Git 명령어로 버전을 관리할 수 있다.

   `cd <로컬 저장소 디렉토리 이름>`

2. 사용자 설정: 이 로컬 저장소를 수정하는 사람의 정보 등록 -> commit에 기록됨

   `git config [--global] user.name <사용자 이름>`

   `git config [--global] user.email <email 주소>`

3. Clone을 한 경우 원격 저장소 주소는 자동으로 `origin`이란 별명으로 등록되어 있다. 로컬 저장소에 연결된 원격 저장소를 확인하는 방법은

   `git remote -v`

4. 저장소 내의 파일 수정/추가 후 변경 사항 stage하기

   `git add .`

5. Stage한 저장소 전체의 상태를 복원 가능한 *버전*으로 메시지와 함께 저장하기

   `git commit -m <변경 사항 메시지>`

6. 파일 변경/추가 - add(4) - commit(5) 반복 하기

7. 원격 저장소에 그 동안 쌓은 commit들 업로드하기

   `git push origin master`

## 4.3. 기존 로컬 저장 저장소에서 다시 작업하기

1. A 방법이든 B 방법이든 기존에 작업하던 로컬 저장소가 있는데 (자신이든 다른 사람이든) 다른 로컬 저장소에서 push를 해서 현재 로컬 저장소에 없는 commit이 원격 저장소에 있다면 이를 먼저 받고 작업을 재개해야 한다.

   `git pull`

2. 이후 작업을 하면서 4.2의 4~7을 반복하면 된다.



참고 : https://goodgodgd.github.io/ian-lecture/archivers/git-intro

---

참고2
--

## 2. 저장소 내용 수정하기

보통 깃헙에 만든 저장소를 **“원격(Remote) 저장소”**라 하고 이 저장소를 PC에 내려받아 실제 작업이 일어나는 저장소를 **“로컬(Local) 저장소”**라 한다. 깃헙을 활용하는 가장 단순한 흐름은 다음과 같다.

1. `git clone`: 원격 저장소를 복사한 로컬 저장소를 만든다.
2. `git add` : 로컬 저장소에서 작업한 내용을 스테이지(stage)한다.
3. `git commit` : 스테이지한 전체 소스의 상태를 커밋(commit)으로 저장한다.
4. `git push` : 커밋을 원격 저장소로 올린다.

여기서는 `sch-robotics`를 저장소 이름으로 사용한다. 다른 이름으로 만든 경우 그 이름에 맞춰 진행하면 된다.

### git clone

> `git clone <repository_url> <dir_name>` : `repository_url` 주소의 원격 저장소를 복사한다. 기본적으로는 저장소 이름과 같은 디렉토리가 생기고 뒤에 `dir_name`을 지정하면 그 이름으로 디렉토리가 생긴다.

### git configure

> Git에 대한 설정을 변경할 수 있는 명령어. 설정할 수 있는 변수가 수백가지지만 대부분은 사용자 등록 정도만 사용한다.
>
> `git config [--global] user.name <name>` : 사용자 이름을 등록한다. `--global` 옵션을 쓰면 이 PC의 모든 로컬 저장소에 기본 사용자가 된다. `--global`을 빼면 현재 저장소에만 적용이 된다.
>
> `git config [--global] user.email <email>`: 사용자의 이메일을 등록한다. `--global`의 용도는 위와 같다.
>
> `git config [--global] --list` : Git의 모든 설정 정보를 조회한다. `--global`을 쓰면 `--global` 옵션을 주고 설정한 전역 설정 정보를 조회한다.

### git status

> HEAD로부터 변경 사항이 있는 파일들을 상태별로 묶어서 보여준다. Untracked, Not staged, Staged, Changes to be commited 등의 상태가 있다. 그리고 현재 상태에서 쓸만한 명령어 추천해준다.

### git add

> 지정한 파일(들)의 최신 상태를 인덱스에 업데이트한다. Untracked나 Modified 상태의 파일을 stage하여 다음 commit에 변경사항이 들어갈 수 있게 준비한다.
>
> `git add <filename>` : 특정 파일을 stage 한다.
>
> `git add <pattern like *.txt>` : 현재 디렉토리에서 패턴과 일치하는 모든 파일을 stage 한다.
>
> `git add .` or `git add -A`: 현재 디렉토리와 하위 디렉토리의 모든 변경된 파일들을 stage 한다.

### git commit

> 현재 인덱스 상태를 저장소에 저장한다. commit을 하면 stage한 상태까지를 저장하고 hash 혹은 checksum을 부여한다. (checksum과 hash는 비슷하게 쓰이지만 방식이 다르고 git에서는 checksum이 더 정확한 용어지만 hash라고 많이 부른다.) Hash는 코드로부터 자동으로 생성되는 40자리 문자로서 commit의 ID 같은 역할을 한다. Hash는 `git log` 명령어를 통해 확인할 수 있다. 이 hash를 이용해 나중에 언제든 예전에 commit한 상태로 돌아갈 수 있다.
>
> `git commit -m <message>` : 현재 stage된 변경사항을 message, author, hash와 함께 저장한다. `-m` 옵션이 없으면 Git 기본 에디터가 실행되서 그곳에서 메시지를 작성하게 한다. 그냥 커맨드에서 -m 옵션을 써서 메시지를 입력하는게 낫다.
>
> `git commit --amend -m <message>` : 직전 commit에서 빠진게 있을 때 변경 사항을 추가하고 add 한 뒤 `--amend` 옵션을 붙여 commit하면 직전 commit을 없애고 추가 변경 사항까지 합친 새로운 commit을 만든다. 변경 사항이 없더라도 단순히 commit message를 다시 쓰고자 할 때도 사용된다.

### git push

> 로컬 저장소의 새로운 commit을 원격 저장소로 올린다. 현재 로컬 저장소의 파일 상태나 Stage 여부에 상관없이 오직 commit에 들어간 변경 사항만 원격 저장소로 올린다.
>
> `git push origin main` : Git 초보자들이 가장 많이 쓰는 명령어 중 하나이다. 저장소를 clone 받으면 `main`라는 기본 브랜치(branch)가 선택되고 원격 저장소는 자동으로 `origin`이란 이름으로 저장된다. 그래서 `main` 브랜치에서 작업 후 commit을 원격 저장소로 업로드 할 때 이 명령어를 쓰게된다.
>
> `git push <remote_repository> <local_branch>` : local_branch의 commit들을 원격 저장소의 같은 이름의 브랜치에 올린다. 예를 들어 `git push origin main`는 로컬 저장소의 `main` 브랜치에 쌓인 commit들을 원격 저장소의 `main` 브랜치(origin/main)에 올린다는 것이다. 원격 저장소에 local_branch가 없을 경우 GitHub에서 자동으로 같은 이름의 브랜치를 만들어준다.
>
> `git push <remote_repository> <local_branch>:<remote_branch>` : local_branch의 commit들을 원격 저장소의 remote_branch에 반영한다.
>
> `git push --all` : 모든 로컬 브랜치를 한번에 push 한다.

### git diff

> diff는 두 상태의 차이를 보여주는 verb다. 변경 사항을 확인할 때 매우 유용하다.
>
> `git diff` : 작업 트리와 인덱스의 차이점 보기 (Unstaged 변경 사항 보기)
>
> `git diff --cached` : 인덱스와 저장소(HEAD)의 차이점 보기 (Staged 변경 사항 보기)
>
> `git diff HEAD` : 작업 트리와 저장소(HEAD)의 차이점 보기
>
> `git diff <start> [end]` : start로부터의 변경 사항이나 start와 end 사이의 변경 사항을 본다. start와 end는 commit hash나 HEAD~n, 브랜치 명, 태그 명이 될 수 있다.

`git diff`를 통해서 변경사항을 확인할 수 있다. `git add` 하지 않은 변경사항은 `git diff`로 보고 `git add`가 된 변경사항은 `git diff --cached`로 확인할 수 있다. 변경사항 확인 후 커밋까지 한다.

### git rm

> rm은 remove의 약자로 파일을 삭제하고 삭제한 상태를 stage한다. 즉 파일을 삭제한 후 `git add .` 한 것과 같다. Git으로 버전 관리되는 파일은 가급적 rm을 이용해 삭제하는 것이 좋다.
>
> `git rm <file_name>` : 지정한 파일을 삭제하고 삭제한 상태를 stage한다.
>
> `git rm <file_pattern>` : 패턴과 일치하는 모든 파일을 삭제하고 stage한다. 예를들어 `git rm *.txt` 라고 하면 모든 텍스트 파일을 삭제하는 것이다.
>
> `git rm -rf <dir_name>` : 디렉토리를 삭제할 때는 -rf 옵션을 줘야한다.
>
> `git rm --cached <file_name>` : 실제 파일은 삭제하지 않고 파일을 인덱스에서 제외하여 Untracked 상태로 만든다.

### .gitignore

하지만 삭제를 하더라도 또다시 `python3 anything.py`를 실행하면 pyc 파일이 다시 생길 것이다. 매번 삭제할 수는 없으므로 git에서 이런 파일들을 **무시**하도록 해야한다. 이때 무시해야할 파일(이나 디렉토리)의 이름(이나 패턴)의 목록을 `.gitignore`라는 파일에 저장한다.

```
~/workspace/robotics-home$ gedit .gitignore
# 파일 내용 작성 후 닫기
__pycache__
*.pyc

~/workspace/robotics-home$ python3 anything.py
~/workspace/robotics-home$ ls
~/workspace/robotics-home$ git status
...
Changes to be committed:
  (use "git reset HEAD <file>..." to unstage)

	deleted:    __pycache__/add_lists.cpython-36.pyc

Untracked files:
  (use "git add <file>..." to include in what will be committed)

	.gitignore
```

### git pull

> 원격 저장소의 새로운 변경 사항(commit)들을 로컬 저장소에 내려받고 작업 트리에 그 내용을 반영한다. Pull을 실행하기 전에 반드시 로컬 저장소의 상태는 모든 것이 commit이 된 “Unmodified” 상태여야 pull을 할 수 있다. Pull은 사실 모든 commit을 내려받는 `git fetch`와 내려받은 commit들과 현재 로컬 파일에 반영하는 (합치는) `git merge FETCH_HEAD` 두 명령어를 결합한 것이다. 따라서 pull에는 merge와 관련된 옵션들이 있다.
>
> `git pull` : 원격 저장소의 모든 브랜치의 commit들을 로컬 저장소에 받고 각 브랜치를 모두 merge 한다. 원격의 main은 로컬의 main와 합치고 원격의 some_branch는 로컬의 some_branch와 합친다.
>
> `git pull <remote> <local_branch>` : 특정 local_branch만 변경 사항(commit)을 내려받고 합친다.
>
> `git pull [--ff / --no-ff / --only-ff]` : merge 방식에 fast-forward 방식과 non-fast-forward 방식이 있는데 두 방식에 대한 설명은 [이곳](https://backlog.com/git-tutorial/kr/stepup/stepup1_4.html)에서 확인할 수 있다. `--only-ff`는 fast-forward 방식이 가능할 때만 merge를 하라는 것이다.

### git mv

> mv는 move의 약자로 파일을 이동하고 이동한 상태를 stage한다. 버전 관리되는 파일을 A 디렉토리에서 B 디렉토리로 옮겨버리면 A에서는 파일이 삭제되고 B에는 새 파일이 추가된 것으로 인식한다. 이를 stage하기 위해서는 역시 `git add -A`를 해줘야한다. 그러므로 파일을 이동할 때는 가급적 mv를 이용하는 것이 좋다.
>
> mv는 또한 단순히 파일의 이름을 바꾸는데도 사용된다. 같은 경로에 다른 이름으로 옮기면 파일명 변경이 된다.
>
> `git mv <src_file> <dst_file>` : src_file을 dst_file로 이름을 바꾼다.
>
> `git mv <src_file> <dst_dir>` : src_file을 dst_dir 디렉토리로 옮긴다.
>
> `git mv <src_file> <dst_dir/dst_file>` : src_file을 dst_dir 디렉토리 아래 dst_file이란 이름으로 옮긴다.

Git 저장소 안에서 파일은 그림처럼 네 가지 상태를 가질 수 있다.

- **Untracked**: 아직 Git으로 버전 관리되지 않은 상태이다.
- **Staged**: `git add`를 실행하면 *Untracked*나 *Modified* 상태의 파일들이 *Staged*가 된다. Stage 했을 당시의 파일 상태가 이후 commit할 때 저장이 된다. 그래서 add를 한 후에 수정을 하더라도 commit은 현재 상태가 아닌 add 한 당시의 상태를 저장한다.
- **Modified**: Stage 하지 않은 변경 사항이 있는 상태다. *Staged* 상태의 파일을 수정하면 Stage 이후의 변경 사항은 *Modified*에 속해서 하나의 파일이 *Staged* 이면서 동시에 *Modified* 상태가 된다. *Unmodified* 파일을 수정하면 완전히 *Modified* 상태가 된다.
- **Unmodified**: *Staged* 상태의 파일을 commit 하면 *Unmodified* 상태가 된다. Git은 commit 단위로 해쉬(hash)를 할당하고 버전을 관리한다. 즉 언제든 과거에 commit한 상태로 돌아갈 수 있다.

원격 저장소와 교류하는 동작(verb)은 네 가지가 있다.

- **pull**: 원격 저장소의 commit을 로컬 저장소에 다운로드 받고 내용을 합친다.
- **push**: 로컬 저장소의 commit을 원격 저장소로 올려서 합친다. 로컬 저장소에 없는 commit이 원격 저장소에 있다면 먼저 pull을 실행해야 push 할 수 있다.
- **clone**: 원격 저장소를 복사한 로컬 저장소를 만든다.
- **fetch**: pull은 사실 원격 저장소의 commit을 내려받는 fetch와 내용을 합치는 merge 두 명령을 한번에 실행한 것이다. fetch만 하게 되면 현재 로컬 저장소의 파일에는 변화가 생기지 않는다.

- **작업 트리, Work Tree**: 현재 사용자에게 보이는 파일들이 있는 공간이다. 이곳에는 다양한 상태의 파일들이 섞여있다.
- **인덱스, Index**: 가장 최근 commit에 Stage 한 내용까지 반영된 공간이다. 파일을 add 하면 그 당시의 상태가 이곳으로 복사된다. commit 하기 전까지 add 한 변경 사항이 이곳에 쌓인다.
- **저장소, Repository**: commit을 하게 되면 Staged 파일들이 새로운 commit으로 저장되고 HEAD가 새 commit으로 변경된다. 저장소에는 commit들이 쌓인다.
- **HEAD**: 그림에서 저장소 대신 HEAD가 쓰이기도 하는데 HEAD는 현재 작업하는 로컬 브랜치의 최신 commit을 가리키는 포인터다.

# GitHub and Branch

## 1. 브랜치(Branch) 개념

브랜치는 여러 사람이 협업하는데 있어서 필수적인 기법이다. 지금까지 실습한 내용은 모두 `main`라는 메인 브랜치에서만 작업을 한 것이다. 하지만 여러 사람이 하나의 브랜치에서 동시에 작업을 하게 되면 여러 문제가 발생할 것이다. 여러 사람이 작업중에 누군가 완성되지 않은 코드를 원격 저장소에 올리고 그걸 다른 작업중인 사람들이 받게되면 에러가 날 수도 있고 동작이 달라질 수 있다. 다른 사람이 작업중에 불필요한 영향을 많이 받아 작업 효율이 크게 저하된다.

브랜치를 쓰면 이러한 문제를 해결할 수 있다. 브랜치는 메인 브랜치의 특정 버전에서 분기(branching)하여 새로운 기능을 넣거나 이슈를 해결하는 등의 하나의 **작업 단위**를 진행하며 자유롭게 commit할 수 있는 **독립적인 작업공간**이다. 자신의 브랜치를 만들어 그곳에서 작업하는 동안에는 남의 눈치를 보지 않고 마음껏 코딩을 해도 된다. 기능이 어느정도 완성되면 코드 정리와 동작 테스트를 한 후 다른 사람들의 동의를 얻어 자신이 만든 변경사항을 메인 브랜치에 합친다. 그리고 다시 새로운 브랜치를 만들어 새로운 기능을 만들거나 이슈를 해결한다. 이것이 일반적인 git을 활용한 작업 흐름이다.

각 개발자는 자신이 단위 기능을 만드는 동안에는 다른 사람의 영향을 받지 않고 새로운 기능에만 집중할 수 있다. 프로젝트 관리 측면에서는 메인 브랜치를 작업 중인 브랜치와 분리하여 메인 브랜치의 안정성을 보장할 수 있다. 메인 브랜치를 업데이트 할 때는 작업 중간 상태가 아닌 작업이 완료된 상태의 코드를 보고 판단하면 되므로 프로젝트 관리를 효율적으로 할 수 있다.

커밋과 브랜치를 비교해보면, 커밋(commit)은 작은 단위의 변경사항을 짧은 메시지와 함께 기록한다. 브랜치는 새로운 기능이나 특정 이슈 해결이라는 최소단위의 토픽(topic)을 잡고 브랜치 내부에서 여러 커밋을 쌓아서 그 기능을 완성한다. 토픽 브랜치를 메인 브랜치에 합친다는 것은 토픽 브랜치의 커밋을 메인 브랜치에 추가한다는 것이다. 그래서 커밋은 작은 단위(함수, 클래스)의 변경 사항이고 브랜치는 기능, 이슈 단위의 변경 사항이라고 볼 수 있다.

## 2. 브랜치 만들어 작업하기

먼저 기본적인 브랜치를 다루면 명령어부터 익혀본다. `git branch`는 일반적인 브랜치 관리 명령어고 `git checkout`은 브랜치를 전환하는데 쓴다.

### git branch

> Git을 이용한 협업의 핵심인 브랜치를 관리하는 verb다.
>
> `git branch` : 로컬 저장소의 브랜치 목록을 보여준다.
>
> `git branch -r` : 원격 저장소의 브랜치 목록을 보여준다.
>
> `git branch <branch_name>` : 새로운 브랜치를 만든다.
>
> `git branch -m <old_name> <new_name>` : 브랜치의 이름을 변경한다.
>
> `git branch -d <branch_name>` : 브랜치를 삭제한다. HEAD에 병합되지 않은 브랜치를 삭제하려면 (즉 브랜치의 commit을 영구적으로 삭제하려면) `-D` 옵션을 준다.

### git checkout

> HEAD를 다른 commit 혹은 브랜치로 옮기고 작업 트리를 그 commit의 snapshot으로 복원한다. 목적지를 다른 브랜치로 지정하면 그 브랜치의 최신 commit으로 HEAD가 옮겨지고 작업 트리가 바뀐다. 그래서 주로 작업 브랜치를 변경하는데 주로 쓰인다.
>
> `git checkout <other_branch>` : other_branch로 작업 브랜치를 바꾸고 작업 트리 영역을 other_branch의 최신 commit 상태로 복원한다.
>
> `git checkout -b <new_branch>` : 현재 상태에서 새로운 브랜치를 생성하고 그곳으로 브랜치를 옮긴다. HEAD의 commit이 변하지 않고 단지 브랜치만 바뀐다. 그래서 작업 트리도 변하지 않는다.
>
> `git checkout HEAD -- <filename>` : 파일의 상태를 HEAD (최신 commit)으로 복원하는 명령어다. 잘못된 변경 사항이 있을 때 주로 쓴다.

```
$ cd ~/workspace/robotics-schl
# 브랜치 목록 보기, 현재 브랜치 확인
~/workspace/robotics-schl$ git branch
# 'new-feature'라는 브랜치 만들기
~/workspace/robotics-schl$ git branch new-feature
~/workspace/robotics-schl$ git branch
# 'new-feature'로 브랜치 전환하기
~/workspace/robotics-schl$ git checkout new-feature
~/workspace/robotics-schl$ git branch
# 'new-feature'를 'new-topic'으로 이름 바꾸기
~/workspace/robotics-schl$ git branch -m new-feature new-topic
~/workspace/robotics-schl$ git branch
# 'main'으로 브랜치 전환하기
~/workspace/robotics-schl$ git checkout main
~/workspace/robotics-schl$ git branch
# 'new-topic'로 브랜치 삭제하기
~/workspace/robotics-schl$ git branch -D new-topic
~/workspace/robotics-schl$ git branch
# 브랜치 만들고 만든 브랜치로 전환 한 줄에 처리
~/workspace/robotics-schl$ git checkout -b make-list-operators
```

### git merge

> `git merge <other_branch>` : 현재 branch에서 other_branch의 commit들을 병합한다.
>
> `git merge [--ff / --no-ff / --ff-only] <other_branch>` : 병합하는 방식을 지정한다. 기본은 –ff (fast-forward)인데 –no-f (non-fast-forward)와의 차이는 [이곳](https://backlog.com/git-tutorial/kr/stepup/stepup1_4.html)에서 보는 것이 좋다. `--only-ff`는 fast-forward 방식이 가능할 때만 merge를 하라는 것이다.
>
> `git merge --squash <other_branch>` : other_branch의 모든 commit들을 하나의 commit으로 합쳐서 병합한다.

`git merge`는 두 개의 브랜치를 합칠 때 쓴다. 메인 브랜치로 전환 후 토픽 브랜치인 `make-list-operators` 브랜치를 병합한다. 병합 전에 `git pull`을 실행하여 다른 사람이 올린 변경 사항을 모두 병합 후 내가 만든 토픽 브랜치를 합쳐야 코드를 최신상태로 업데이트 할 수 있다.



## 3. Pull Request 만들기

그래서 팀으로 일하는 개발자들은 브랜치를 합칠 때 자신이 임의로 합치지 않고 **Pull Request (PR)**라는 문서를 작성하여 저장소 관리자나 동료들에게 공유하고 동의를 구한다. git에서 “pull”이란 다른 저장소의 변경사항을 내 저장소에 반영한다는 의미가 있으므로 “Pull Request”란 내가 만든 브랜치의 변경사항을 원격의 메인 브랜치에 반영할 수 있도록 요청한다는 뜻이다.

### GitHub에 PR 작성하기

이제 작업한 브랜치를 원격 저장소로 올린다. 바로 `main` 브랜치에 합치지 않고 원격 저장소에 브랜치를 Pull Request와 함께 올려서 내가 한 일을 정리하고 다른 사람의 리뷰를 받을 수 있게 하는 것이다.

원격 저장소로 브랜치를 올리려면 `git push origin <branch_name>` 명령어를 쓴다. `origin`은 `git clone`을 할 때 자동지정된 원격 저장소의 이름인데 `git remote -v` 명령어를 통해 `origin`의 실제 주소를 볼 수 있다.

```
~/workspace/robotics-schl$ git remote -v
origin	https://github.com/goodgodgd/sch-robotics.git (fetch)
origin	https://github.com/goodgodgd/sch-robotics.git (push)


~/workspace/robotics-schl$ git push origin make-dict-operators
~ git 홈페이지 가서 확인

~/workspace/robotics-schl$ git checkout main
~/workspace/robotics-schl$ git pull origin main
```



---



Git - 기타
--



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

참고

http://yoonbumtae.com/?p=2366



---



#### 태그 조회하기

태그를 전체를 조회할 때는 `git tag`를 사용하여 조회합니다.

```
# git tag
v1.0.0
v1.0.1
v1.1.0
```

만약 원하는 태그명을 조건으로 검색하고자 한다면 `git tag -l v1.1.*`과 같이 사용합니다.

```
# git tag -l v1.1.*
v1.1.0
```

#### 태그 붙이기

태그는 Lightweight와 Annotated 두 종류가 있습니다. Lightweight 태그는 특정 커밋을 가르키는 역할만 합니다. 한편 Annotated 태그는 태그를 만든 사람, 이메일, 날짜, 메시지를 저장합니다. 그리고 [GPG(GNU Privacy Guard)](http://ko.wikipedia.org/wiki/GNU_프라이버시_가드)로 서명할 수도 있습니다.

Lightweight 태그는 `git tag [Tag Name]`으로 붙일 수 있습니다.

```
# git tag v1.0.2
# git tag
v1.0.2
```

Annotated 태그는 `-a` 옵션을 사용합니다.

```
# git tag -a v1.0.3 -m"Release version 1.0.3"
```

`git show v1.0.3`을 통해 태그 메시지와 커밋을 확인할 수 있습니다.

```
# git show v1.0.3

tag v1.0.3
Tagger: minsOne <cancoffee7+github@gmail.com>
Date:   Sat Feb 15 17:53:49 2014 +0900

Release version 1.0.3

commit 4bb37290cb55490a9829b4ff015b340d513f132a
Merge: e0d819c 12aa1b0
Author: Markus Olsson <j.markus.olsson@gmail.com>
Date:   Thu Feb 13 15:26:47 2014 +0100

    Merge pull request #947 from IonicaBizau/patch-1
    
    Updated the year :-)
```

태그를 이전 커밋에 붙여야 한다면 커밋 해쉬를 추가하여 사용할수 있습니다.

```
# git tag v1.0.5 03c0beb080

# git tag -a v1.0.4 -m"Release version 1.0.4" 432f6ed

# git tag
v1.0.4
v1.0.5

# git show v1.0.4

tag v1.0.4
Tagger: minsOne <cancoffee7+github@gmail.com>
Date:   Sat Feb 15 18:02:02 2014 +0900

Release version 1.0.4

commit 432f6edf3876a5e2aa8ea545fd15f99953339aba
Author: Denis Grinyuk <denis.grinyuk@gmail.com>
Date:   Mon Feb 3 14:52:36 2014 +0400

    Additional comments
```

만약 GPG 서명이 있다면 `-s` 옵션을 사용하여 태그에 서명할 수 있습니다.

```
# git tag -s v1.0.3 -m"Release version 1.0.3"
```

#### 태그 원격 저장소에 올리기

태그를 만들고 원격 저장소에 올려야할 필요가 있다면 브랜치를 올리는 방법과 같이 사용할 수 있습니다.

```
# git push origin v1.0.3
```

모든 태그를 올리려면 `--tags`를 사용합니다.

```
# git push origin --tags
```

#### 태그 삭제하기

필요없거나 잘못 만든 태그를 삭제하기 위해선 `-d`옵션을 사용하여 삭제할 수 있습니다.

```
# git tag -d v1.0.0
```

원격 저장소에 올라간 태그를 삭제하기 위해선 `:`를 사용하여 삭제할 수 있습니다.

```
# git push origin :v1.0.0
```



참고 : http://minsone.github.io/git/git-addtion-and-modified-delete-tag

---

Git 토큰 활성화 

1. git 홈페이지 로그인
2. setting
3. developer settings
4. personal access tokens
5. 



참고 : https://velog.io/@jakezo/GitHub-%ED%86%A0%ED%81%B0-%EC%9D%B8%EC%A6%9D-%EB%A1%9C%EA%B7%B8%EC%9D%B8-Personal-Access-Token-%EC%83%9D%EC%84%B1-%EB%B0%8F-%EC%82%AC%EC%9A%A9-%EB%B0%A9%EB%B2%95