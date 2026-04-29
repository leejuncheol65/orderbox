# QJIC Safe BabyAI

QJIC Safe BabyAI는 **온디바이스 보이스피싱/위험문자 자가학습 AI** 앱입니다.
실제 수신 SMS 또는 직접 입력 문구를 대상으로 정상/의심/위험 3단계 판정을 수행하고,
사용자 피드백으로 개인화 가중치(W_personal)를 학습합니다.

## 핵심 특성
- 3층 판단 구조: `W_base + W_adapter + W_personal`
- 잔여 신호 학습: `I = S + L`
- Anchor Memory 기반 `L_old` 모니터링 및 약한 replay 보정
- 모든 판정/학습은 휴대폰 내부에서만 수행
- 인터넷 전송 없음
- 문자 내용 서버 전송 없음
- 내부 저장(SharedPreferences)만 수행

## APK 빌드 방법
### 터미널 (Linux/macOS)
```bash
./gradlew assembleDebug
```

### 터미널 (Windows)
```bat
.\gradlew.bat assembleDebug
```

### Android Studio에서 빌드
1. Android Studio에서 프로젝트 열기
2. 상단 메뉴 **Build > Build APK(s)** 선택
3. 빌드 완료 후 알림에서 APK 위치 확인

## APK 출력 위치
`app/build/outputs/apk/debug/app-debug.apk`

## SMS 권한 및 정책 메모
- 이 앱은 `RECEIVE_SMS`, `POST_NOTIFICATIONS` 권한을 사용합니다.
- 본 프로젝트는 **개인 테스트 APK 기준**입니다.
- 구글플레이 출시 시에는 SMS 관련 권한 정책, 기본 SMS 앱 등록 요건 또는 대체 정책 대응이 별도로 필요합니다.

## 개인정보/네트워크 정책
- 인터넷 권한을 선언하지 않습니다.
- SMS 내용은 서버로 전송하지 않습니다.
- 모든 데이터는 로컬 내부 저장소(SharedPreferences)에만 저장됩니다.
