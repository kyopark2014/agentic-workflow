# Nova Pro로 Agentic Workflow 활용하기

<p align="left">
    <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fkyopark2014%2Flanggraph-nova&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false")](https://hits.seeyoufarm.com"/></a>
    <img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green">
</p>


여기서는 LangGraph로 구현한 agentic workflow를 구현하고 [Streamlit](https://streamlit.io/)을 이용해 개발 및 테스트 환경을 제공합니다.  

한번에 배포하고 바로 활용할 수 있도록 [AWS CDK](https://docs.aws.amazon.com/cdk/api/v2/docs/aws-construct-library.html)를 이용하고 ALB - EC2의 구조를 이용해 scale out도 구현할 수 있습니다. 또한, [CloudFront](https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/Introduction.html) - ALB 구조를 이용해 배포후 바로 HTTPS로 접속할 수 있습니다. Streamlit이 설치되는 EC2의 OS는 EKS/ECS와 같은 컨테이너 서비스에 주로 사용되는 [Amazon Linux](https://docs.aws.amazon.com/linux/al2023/ug/what-is-amazon-linux.html)를 base하여, 추후 상용으로 전환할 때에 수고를 줄일 수 있도록 하였습니다.

## System Architecture 

이때의 architecture는 아래와 같습니다. 여기서에서는 streamlit이 설치된 EC2는 private subnet에 둬서 안전하게 관리합니다. [Amazon S3는 Gateway Endpoint](https://docs.aws.amazon.com/vpc/latest/privatelink/vpc-endpoints-s3.html)를 이용하여 연결하고 Bedrock은 [Private link](https://docs.aws.amazon.com/ko_kr/bedrock/latest/userguide/usingVPC.html)를 이용하여 연결하였으므로 EC2의 트래픽은 외부로 나가지 않고 AWS 내부에서 처리가 됩니다. 인터넷 및 날씨의 검색 API는 외부 서비스 공급자의 API를 이용하므로 [NAT gateway](https://docs.aws.amazon.com/vpc/latest/userguide/vpc-nat-gateway.html)를 이용하여 연결하였습니다.

<img width="800" alt="image" src="https://github.com/user-attachments/assets/2fd4a38c-170c-401a-8103-e5641ed25378" />

## 상세 구현

Agentic workflow (tool use)는 아래와 같이 구현할 수 있습니다. 상세한 내용은 [chat.py](./application/chat.py)을 참조합니다.


### 활용 방법

EC2는 Private Subnet에 있으므로 SSL로 접속할 수 없습니다. 따라서, [Console-EC2](https://us-west-2.console.aws.amazon.com/ec2/home?region=us-west-2#Instances:)에 접속하여 "app-for-llm-streamlit"를 선택한 후에 Connect에서 sesseion manager를 선택하여 접속합니다. 

Github에서 app에 대한 코드를 업데이트 하였다면, session manager에 접속하여 아래 명령어로 업데이트 합니다. 

```text
sudo runuser -l ec2-user -c 'cd /home/ec2-user/langgraph-nova && git pull'
```

Streamlit의 재시작이 필요하다면 아래 명령어로 service를 stop/start 시키고 동작을 확인할 수 있습니다.

```text
sudo systemctl stop streamlit
sudo systemctl start streamlit
sudo systemctl status streamlit -l
```

Local에서 디버깅을 빠르게 진행하고 싶다면 [Local에서 실행하기](https://github.com/kyopark2014/llm-streamlit/blob/main/deployment.md#local%EC%97%90%EC%84%9C-%EC%8B%A4%ED%96%89%ED%95%98%EA%B8%B0)에 따라서 Local에 필요한 패키지와 환경변수를 업데이트 합니다. 이후 아래 명령어서 실행합니다.

```text
streamlit run application/app.py
```

EC2에서 debug을 하면서 개발할때 사용하는 명령어입니다.

```text
sudo runuser -l ec2-user -c "/home/ec2-user/.local/bin/streamlit run /home/ec2-user/langgraph-nova/application/app.py"
```



## 직접 실습 해보기

### 사전 준비 사항

이 솔루션을 사용하기 위해서는 사전에 아래와 같은 준비가 되어야 합니다.

- [AWS Account 생성](https://repost.aws/ko/knowledge-center/create-and-activate-aws-account)에 따라 계정을 준비합니다.

### CDK를 이용한 인프라 설치

본 실습에서는 us-west-2 리전을 사용합니다. [인프라 설치](./deployment.md)에 따라 CDK로 인프라 설치를 진행합니다. 

## 실행 결과

### RAG (Knowledge Base)

메뉴에서 RAG를 선택하고, "AWS의 스토리지 서비스에 대해 설명해주세요."라고 입력 후 결과를 확인합니다.

![image](https://github.com/user-attachments/assets/89dbe5f3-0dd1-4829-af80-a5fc51ad03e7)

이번에는 "Bedrock Agent와 S3를 비교해 주세요" 라고 입력후에 결과를 확인합니다. RAG만 적용한 경우에는 사용자의 질문을 그대로 검색하는데, 정확히 관련된 문서가 없으면 적절히 답변할 수 없습니다. 이 문제는 agent를 이용하였을때 query decompostion으로 해결할 수 있습니다. 

![image](https://github.com/user-attachments/assets/a365357a-aaec-4745-ab74-fc3bcb769873)


### Agentic Workflow

Agentic Workflow(Tool Use) 메뉴를 선택하여 오늘 서울의 날씨에 대해 질문을 하면, 아래와 같이 입력하고 결과를 확인합니다. LangGraph로 구현된 Tool Use 패턴의 agent는 날씨에 대한 요청이 올 경우에 openweathermap의 API를 요청해 날씨정보를 조회하여 활용할 수 있습니다. 

![image](https://github.com/user-attachments/assets/4693c1ff-b7e9-43f5-b7b7-af354b572f07)

아래와 같은 질문은 LLM이 가지고 있지 않은 정보이므로, 인터넷 검색을 수행하고 그 결과로 아래와 같은 답변을 얻었습니다.

![image](https://github.com/user-attachments/assets/8f8d2e94-8be1-4b75-8795-4db9a8fa340f)

RAG를 테스트 하였을때에 사용한 "Bedrock Agent와 S3를 비교해 주세요."라고 질문을 하면, 이번에는 좀더 나은 답변을 얻었습니다. 

![noname](https://github.com/user-attachments/assets/969e9b84-5b80-4948-8627-f86bd2af26bc)



### Reference 

