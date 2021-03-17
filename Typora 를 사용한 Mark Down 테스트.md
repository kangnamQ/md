# Typora 를 사용한 **Mark Down** 테스트



- ###### 작성자 : 강남규

- ###### 작성일자 : 21.03.15

---



1. Headers
   ---

   1.  큰제목 : 글을 적은 뒤 Shift+Enter 한 후 === 를 작성
    2.  부제목 : 글을 적은 뒤 Shift+Enter 한 후 --- (3개이상) 작성
    3.  글머리 기호 : # 갯수에 따라 H1 ~ H6까지 지원
       #이 가장크며 ######이 가장 작고 여기까지 지원됨.
       (H7, #######은 지원되지 않음.)



2. BlockQuote
   ---

   ">"을 이용하여 인용 또는 블럭 인용 문자를 사용한다.

   > 사용하면 이런식으로 들어가진다.
   >
   > > 두번쓰면 이런식으로 인용 인용이 가능하다.
   > >
   > > > > > 이런식으로 단을 나눠서 할수있으며
   > > >
   > > > 엔터를 누르면 상위 단으로 나올 수 있다.
   >>
   > > 참고로 여기서 다른 요소도 사용가능하다. 
   >
   > ***이런식으로***



 3. list(목록)
    ---

    1. 리스트는 1. 2. 3. 같이 사용 할 수 있다.
       - 숫자로 하면 내림차순으로 정의된다.
    2. 글머리 기호 *, +, - 를 지원한다.
       * 세 문자가 동일하게
         + 동일한 점기호를
           - 나오게 한다.

    * 나갈때는 동일하게 엔터로 나올 수 있다.
      * 추가적으로 나오고 싶거나 할때는 ***Shift + Tab***을 누르면 앞단으로 간다.
        * 이것도 탭기준으로 글머리기호가 바뀌거나 한다.

​	

4. **Code (코드)** 
   ---

   1. 들여쓰기

      > Python 같은 느낌으로 1탭 혹은 4개의 스페이스(공백)으로 들여쓰기 된 부분을
      >
      > 만나면 변환이 된다.
      >
      > > 
      > >
      > > ​	print("Hello")
      > >
      > >  
      >
      > 사용할 때는 한 줄을 띄어써야 인식이 제대로 된다고 한다.
      >
      > 띄어쓰지 않으면 인식이 제대로 안되서 그냥 일자로 쭉 나온다.

      ​	print("Hello World")

   

   근데 된다고는 하는데... 안나오는 것 같기도 하고... 잘 모르겠다



5. 코드 블럭
   ---

   * 참고로 방금알아낸건데 저런 '*', '**' 으로 변환하는 건 붙여써야 하고 
   * #이런건 띄어써야 적용이 된다... 아마 쓰다보면 정확히 알 수 있을 것 같다.

   

   먼저 코드블럭을 이용하는 방법은 두가지가 있다.

   * 코드를 변화시키지 않고 쓰고 싶을때는 ` 으로 감싸서 사용하면 된다.
   * `는 숫자 1왼쪽에 있는 ~ 쓸 때 사용하는 키이다.

   1. ` <pre> <code>{code}</code></pre>`를 사용하는 방식

      ```
      	<pre>
      	    <code>
      	    Code {
      	    	code
      	    }
      	    </code>
      	</pre>
      ```
   
2.  코드블럭 코드 ("```")를 이용하는 방법 
   
   ``` Hello World```
   
   - 이때 시작접에 사용하는 언어를 선언해서 문법강조가 가능하다.
   
     ``` python
        print("Hello World")
        '''
        문법강조를 이용하면 코딩하는 것과 같이 색이 변하고 강조된다.
        '''
     ```
   
     두가지 방법 모두 코드블럭이 나오니 편한것을 사용하면 되겠다.



6. 수평선
   ---

   ` ***, * * *, *****` 나  `---, - - -, ------------------------`을 사용하면 수평선을 만들 수 있다.

   글 중간 중간을 나눌때에 편리하다.



7. 링크
   ---

   - **참조링크**

   ```
   [link keyword][id]
   
   [id]: URL "Opthinal Title here"
   
   //code
   Link : [Google][googlelink]
   [Googlelink]: https://google.com "Go google"
   ```

   Link : [Google][googlelink]

   [Googlelink]: https://google.com "Go google"

   

   - **외부링크**

     ``````
     사용문법: [Title](link)
     
     ex : 
     [Google](https://google.com, "google link")
     ``````

     [Google](https://google.com, "google link")

     

   - **자동연결**

     ``````
     일반적인 URL 혹은 이메일 주소인 경우, 적절한 형식으로 링크를 형성
     
     - 외부링크: <https://google.com/>
     - 이메일 링크: <address@example.com>
     ``````

     - 외부링크: <https://google.com/>
     - 이메일 링크: <address@example.com>



 8. 강조
    ---

    ``````
    *single asterisks*
    _single underscores_
    **double asterisks**
    __double underscores__
    ~~cancelling~~
    ``````

    *single asterisks*
    _single underscores_
    **double asterisks**
    __double underscores__
    ~~cancelling~~

    > `문장 중간에 사용할 경우 **띄어쓰기** 를 사용하는 것이 좋다.`
    >
    > 문장 중간에 사용할 경우  **띄어쓰기** 를 사용하는 것이 좋다.



 9. 이미지
    ---

    ``````
    ![Alt text](/path/to/img.jpg)
    ![Alt text](/path/to/img.jpg "Optional title")
    
    사이즈 조절 기능은 없기 때문에 
    <img width="" height=""></img>를 사용한다.
    
    ex: 
    <img src="path/to/img.jpg" width="450px" height="300px" title="px(픽셀) 크기설정" alt="RubberDuck"></img><br/>
    
    <img src="path/to/img.jpg" width="40%" height="30%" title="px(픽셀) 크기설정" alt="RubberDuck"></img>
    ``````



 10. 줄바꿈
     ---

     3칸이상 띄어쓰기(`   ` )를 하면 줄이 바뀐다.    

     ``````
     * 줄 바꿈을 하기 위해서는 문장 마지막에서 3칸이상을 띄어쓰기 해야한다.
     이렇게
     
     * 줄 바꿈을 하기 위해서는 문장 마지막에서 3칸이상을 띄어쓰기 해야한다.___\\ 띄어쓰기
     이렇게
     ``````

     * 줄 바꿈을 하기 위해서는 문장 마지막에서 3칸이상을 띄어쓰기 해야한다. 
       
       이렇게
       
     * 줄 바꿈을 하기 위해서는 문장 마지막에서 3칸이상을 띄어쓰기 해야한다.   
       이렇게

---

##### 배웠으니 참조링크도 써보자

* 참조 : https://gist.github.com/ihoneymon/652be052a0727ad59601

  (그냥 링크를 써봄)


- Reference : [Markdown_reference][Markdown_link]

  [Markdown_link]: https://gist.github.com/ihoneymon/652be052a0727ad59601 "Markdown_reference"

  (참조링크형식으로 써봄)    

