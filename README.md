# AeroMind - 강화학습 기반 드론 주행

## 개요
AeroMind(Aero(공중의) + Mind(인공지능))는 강화학습을 활용하여  
드론이 장애물을 회피하면서 최적 경로로 주행하도록 학습시키는 프로젝트입니다.  
시뮬레이션 환경에서 학습된 정책을 실제 드론에 적용하는 Sim-to-Real 방식을 사용합니다.  

---

## 프로젝트 목표
- 강화학습 기반 최적 경로 탐색
- Sim-to-Real 적용을 통한 실제 드론 주행 최적화  
- LiDAR, 카메라 등 센서를 활용한 안정적인 장애물 회피  

---

## 팀원 구성
- 최종윤  
- 나상은  
- 김민준
- 김병욱

---

## 폴더 구조 
```
📂 airsim_drone_project/
│  
├── 📂 envs/                # 환경 세팅 폴더  
│   ├── airsim_drone_env_0  # AirSim 드론 환경 세팅
│  
├── 📂 model/               # 학습 모델 폴더  
│   ├── PPO_train.py        # PPO 학습 코드  
│  
├── .gitignore              # Git 관리 제외 파일  
├── README.md               # 프로젝트 설명 파일  
├── test.py                 # 테스트 코드  
├── main.py                 # 메인 실행 파일  
├── config.py               # 하이퍼파라미터 설정 파일  
├── callbacks.py            # 학습 중 콜백 함수 

```
---

## 설치 방법

---

## 사용 방법

---

## 사용 기술 및 도구
| 카테고리 | 기술 및 도구 |
|----------|-------------|
| **언어** | Python 3 |
| **강화학습** | Stable-Baselines3, PyTorch |
| **시뮬레이션** | Unreal Engine + AirSim |
| **드론 제어** | PX4 + MAVLink |
| **센서 활용** | LiDAR, 카메라, IMU |

---

## 시뮬레이션 및 학습 방식
### **1️. 시뮬레이션 환경**
- Unreal Engine(4.27.2) 기반의 AirSim 시뮬레이터를 사용하여 학습 진행  

### **2️. 강화학습 과정 (커리큘럼 학습)**
1. 기본 주행 학습 → 장애물 없는 환경에서 목표 지점 도달  
2. 장애물 회피 학습 → 장애물을 피하면서 목표 도달  
3. 제한 시간 적용 → 주어진 시간 내 도달하도록 학습  
4. 센서 노이즈 추가 → 실제 환경과 유사한 데이터 학습  
5. 동적 장애물 대응 → 움직이는 장애물(사람) 회피 학습  

### **3️. Sim-to-Real 적용**
- PX4 기반 드론을 사용하여 강화학습 모델을 실제 드론에 적용 
- MAVLink 프로토콜을 통해 실시간으로 드론을 제어
- 센서 데이터를 분석하여 드론이 안정적으로 비행하도록 조정


---

## 실험 결과

---

## 프로젝트 기간 및 작업 관리 

### 기간  
- 2025-xx-xx ~ 2025-xx-xx

### 작업 관리  
- [GitHub Projects](#)와 [Issues](#)를 사용하여 공유

### 주간회의  
- 주간회의를 진행하여 작업 방향이나 코드 고민에 대해 논의하였으며,  
[GitHub Wiki](#)를 사용하여 기록

#### 🔗 회의록 목록  