# 모델명 한국어 능력 평가

## 복잡한 문제로 수능 국어를 선택한 이유

수학 능력 시험의 국어 영역은 LLM 모델의 한국어 능력을 측정하기 좋은 주제입니다. [지문과 선택지-화법과 작문](https://github.com/NomaDamas/KICE_slayer_AI_Korean/blob/master/data/2023_11_KICE.json)은 json포맷으로 문제와 답을 제공하고 있습니다. [2023년 수능의 국어(화법과 작문)의 1등급 컷](https://www.nextplay.kr/news/articleView.html?idxno=4617)은 92점이었습니다. 여기에서는 [LangGraph를 구현한 planning 패턴](https://github.com/kyopark2014/langgraph-agent?tab=readme-ov-file#plan-and-execute)을 이용하여 CoT 방식으로 동작하는 agentic workflow를 구현하였습니다. 결과적으로 Claude 3.5 Sonnet(v2), Nova Pro 순으로 좋은 결과를 얻었습니다. Nova Pro의 결과는 Claude 3.5 Sonnet (v1)보다 약간 좋았고, Claude 3.5 Sonnet보다도 좋은 결과를 보였습니다.

여기서 사용한 planning 패턴의 동작은 아래 activity diagram을 참조합니다.

![image](https://github.com/user-attachments/assets/e81d9a77-1dc4-490a-a8a6-491eea5c15e0)



## 모델별로 수학능력시험 국어 영역 비교

### 시험 방법

아래와 같이 대화 형태로 RAG를 선택합니다.

![image](https://github.com/user-attachments/assets/99051bad-3532-4204-a234-42561a273067)

아래와 같이 1) 사용 모델을 지정하고 2) CSAT evaluator를 enable 합니다. 이후 [Browse files]를 선택하여 [국어 영역 문제](https://github.com/NomaDamas/KICE_slayer_AI_Korean/blob/master/data/2023_11_KICE.json)을 업로드합니다. 

![noname](https://github.com/user-attachments/assets/7ac16a88-b0cf-4ad6-9131-09b2dc6dca15)

이후 자동으로 수능 문제를 풀기 시작하는데, 같은 동작을 2-3회 반복하면서 결과를 모읍니다.

### 시험 결과
 
아래와 같이 가장 좋은 결과는 Claude 3.5 Sonnet(v2)이었고 2번째로 좋은 모델은 Nova Pro입니다. 이 모델은 Claude 3.5 Sonnet (v1)과 동급의 성능을 보여주고 있습니다. Claude 3.5 Haiku도 괜찮은 성적을 얻지만, Nova Lite나 Claude 3.0 Sonnet은 다른 모델대비 낮은 점수를 얻었습니다.

<img src="https://github.com/user-attachments/assets/4b9f0590-f513-4327-87dd-9b81e312c2fc" width="700">

이때의 정답, 오답의 내용은 아래와 같습니다.

#### Claude 3.5 Sonnet (v2)

<img src="https://github.com/user-attachments/assets/d767e581-a0bc-4752-b972-44aaec991ed1" width="600">

#### Nova Pro 

<img src="https://github.com/user-attachments/assets/b62df1ee-2b95-47da-b7fe-d2f7bc04c88d" width="600">

#### Claude 3.5 Haiku

<img src="https://github.com/user-attachments/assets/07b11094-3b16-484b-bb58-e768ebe2250a" width="600">

## Reference

[지문과 선택지-화법과 작문](https://github.com/NomaDamas/KICE_slayer_AI_Korean/blob/master/data/2023_11_KICE.json)

[수능 문제의 경우에 정답이 알려져있고 상세한 해설서](https://m.blog.naver.com/awesome-2030/222931282476)

[2023년 수능의 국어(화법과 작문)의 1등급 컷](https://www.nextplay.kr/news/articleView.html?idxno=4617)
