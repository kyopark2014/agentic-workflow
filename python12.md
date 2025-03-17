# Python 업그레이드

LangGraph Supervisor, Swarm은 Python 3.10이상을 요구합니다. [EC2의 Base Image](https://docs.aws.amazon.com/linux/al2023/ug/python.html)는 Python 3.9이므로 업그레이드가 필요합니다.

## Python 버전 체크

아래와 같이 Python 버전이 3.9.2라면 업그레이드를 검토합니다.

```text
sh-5.2$ python3 --version
Python 3.9.20
```

아래 명령어로 필요한 프로그램을 설치합니다. 

```text
sudo yum groupinstall "Development Tools" -y
sudo yum erase openssl-devel -y
sudo yum install openssl11 openssl11-devel libffi-devel bzip2-devel wget -y
```

## OpenSSL

Python 3.12를 설치하기 위해서는 OpenSSL을 설치하여야 합니다.

```text
cd 
wget https://www.openssl.org/source/openssl-1.1.1t.tar.gz
tar xvf openssl-1.1.1t.tar.gz

cd openssl-1.1.1t/
./config --prefix=/usr/local/ssl --openssldir=/usr/local/ssl shared zlib
make
sudo make install

export LDFLAGS="-L/usr/local/ssl/lib"
export CPPFLAGS="-I/usr/local/ssl/include"
```

## Python 3.12

아래 명령어로 업그레이드 합니다.

```text
cd
wget https://www.python.org/ftp/python/3.12.1/Python-3.12.1.tgz
tar -xf Python-3.12.1.tgz 
cd Python-3.12.1/
./configure --enable-optimizations
./configure --with-openssl=/usr/local/ssl --with-openssl-rpath=auto
nproc
make -j $(nproc)
sudo make altinstall
```

아래와 같이 버전을 확인할 수 있습니다.

```python
sh-5.2$ python3.12 --version
Python 3.12.1
```

<!--
이제 필요한 패키지를 설치합니다.

```text
python3.12 -m pip install langgraph-supervisor
```

아래와 같이 python3로 모두 업데이트 합니다.

```python
sudo rm /usr/bin/python3
sudo ln -s /usr/local/bin/python3.12 python3
```
-->

streamlit 환경에서는 아래와 같이 패키지를 설치합니다.

```text
sudo runuser -l ec2-user -c 'python3.12 -m pip install langgraph-supervisor'
```



## Reference

[[Python3.11] ssl module in Python is not available 해결](https://datamoney.tistory.com/378)

[[EC2] Amazon Linux 2 python3.12 설치방법](https://kodean.tistory.com/43)

[Python in AL2023](https://docs.aws.amazon.com/linux/al2023/ug/python.html)

[How To Install Python 3.10 on Amazon Linux 2](https://computingpost.medium.com/how-to-install-python-3-10-on-amazon-linux-2-43ddcd511784)

[python 3.12 on Amazon Linux 2023](https://repost.aws/questions/QULIsYrNNAQoiy59gkn8h1jg/python-3-12-on-amazon-linux-2023)


