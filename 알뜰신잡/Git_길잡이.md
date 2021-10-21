

GIT
==



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

---



2. 보통 사용하는 순서
--

```
$ git init

$ git remote add origin <.git 주소>
	저장소 주소를 origin이란 이름으로 추가하겠다
$ git remote -v
	원격 저장소 목록 

	이미 사용하고 있는 폴더로 지정하고 싶다면 remote로 설정해주어도 괜찮지만
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

--- 

$ git clone <원격 저장소 주소> (<디렉토리명>)
	상황에 따라 필요하면 사용

# git pull <원격 저장소> <브랜치>
$ git pull origin master
	push한 내용을 받고싶을 때 사용
```



3. git ignore
   --

   ```
   $ gedit .gitignore
   .git이 있는 폴더에 같이 만들어야 함, 만들고 쓰면 됨
   
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
   
   출처: https://kcmschool.com/194 [web sprit]
   ```
   
   
   
4. Branch 사용 
   --

   ```
   $ git branch
   	branch 목록 확인
   
   $ git branch <name>
   	branch 생성
   	
   $ git checkout <branch_name>
   	branch 이동
   
   작업할 떄는 똑같지만 
   original branch는 main이 default 상태
   main에 push하는 것이 아닌 작업하는 branch에 push를 하면
   main에 반영되지 않으며 Pull Requests라는 과정을 거쳐 push하게 됨
   
   
   $ git add .
   $ git commit -m "want msg"
   
   $ git push origin <branch_name>
   	작업한 branch의 이름을 입력해야함
   
   이후 git 홈페이지에 가서 PR을 작성한뒤
   main관리자가 승인을 해주어야 merge가 된다.
   
   branch끼리의 merge도 가능하지만 추천하지 않음
   
   merge가 됬다면
   $ git checkout main
   $ git pull origin main
   
   을 통하여 작업한 데이터를 받아올 수 있음.
   ```

   

---

Git 토큰 활성화 
--

1. git 홈페이지 로그인
2. setting
3. developer settings
4. personal access tokens

   

참고 : https://velog.io/@jakezo/GitHub-%ED%86%A0%ED%81%B0-%EC%9D%B8%EC%A6%9D-%EB%A1%9C%EA%B7%B8%EC%9D%B8-Personal-Access-Token-%EC%83%9D%EC%84%B1-%EB%B0%8F-%EC%82%AC%EC%9A%A9-%EB%B0%A9%EB%B2%95