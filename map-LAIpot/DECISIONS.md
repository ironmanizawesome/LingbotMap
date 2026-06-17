# map-LAIpot — 의사결정 로그 (Decision Log)

> 온실 화분작물 per-pot LAI 트랙을 설계하며 거친 **질문·사고·결정**의 기록.
> 코드가 "무엇"을 하는지는 OVERVIEW.md, "왜 그렇게 됐는지"는 이 문서.

---

## 배경
lingbot-map(GCT 기반 스트리밍 3D 재건)으로 온실 작물(딸기·수박 등)의 **LAI(엽면적지수)**를 추정하려 한다. 촬영은 핸드헬드 가로 이동. 기존 `map-LAIcrop`(노지/수직 촬영, 지면평면 기준)과 가정이 달라 별도 트랙으로 분기.

---

## 1. 촬영 방식 — 수직(nadir) vs 가로 이동
- **맥락**: 군락을 위에서 내려찍기보다, 가로로 이동하며 촬영. 디테일이 더 필요하면 기존 고도보다 조금만 더 위에서.
- **결정**: **가로 이동 촬영**으로 진행. 온실 기준으로 범위 확정.
- **이유**: 군락 측면 정보가 자연스럽게 잡히고, 수직 촬영 장비/동선 부담이 큼. 단 이로 인해 아래 2·3번 문제가 따라옴.

## 2. 단차/이중단차 걱정 → "면적은 높이(offset)에 불변, **normal에만 의존**"
- **사용자 우려**: 베드 위 화분(=floor/베드/화분 흙의 단차·이중단차)이면 RANSAC+LS로도 지면 기준을 못 잡아 LAI가 틀어지지 않나?
- **코드 분석 결론**: 직접법 면적([map-LAIcrop/lai.py](../map-LAIcrop/lai.py))은 점을 **법선에 수직인 u,v로 투영**해 계산 → 평면 **높이 d가 식에서 상쇄**된다. 즉 floor/베드/화분 흙 중 무엇에 평면을 잡든(=d만 다름) **면적·LAI는 동일**.
- **함의**: **단차(높이 차이) 자체는 LAI를 안 망친다.** 진짜로 중요한 건 (a) **법선(중력 up) 방향**, (b) 여러 화분이 한 hull로 묶이는 문제(→ 개체 분리). 그래서 "지면 평면"이 아니라 **"up 벡터 + per-pot"**로 문제를 재구성.

## 3. z축/up은 어디서 오나 → **중력 GT가 없다**
- **질문**: lingbot이 z축(up)을 자동으로 정하나? 그게 중력 GT가 맞나? 인지하고 수정했나?
- **발견**: 월드 프레임은 **첫 번째 카메라 기준**이다. [glb_export.py:311 `apply_scene_alignment`](../lingbot_map/vis/glb_export.py#L311)가 씬을 `inv(extrinsics[0])`로 정렬(VGGT 관례: 첫 프레임이 기준). 뷰어가 "첫 카메라로 정렬"하는 것 자체가 **중력 정렬 GT가 없다는 증거**.
- **함의**:
  - lingbot의 월드 z축은 **중력과 무관**(가로로 들면 임의 방향).
  - 기존 `ground.py`의 "최저 z=지면" 가정은 **수직 촬영 전제** → 가로촬영에선 무효.
  - **수정 작업 = `up_vector.py`의 존재 이유.** 월드 z를 믿지 않고 **카메라 포즈에서 up을 유도**(`up_from_cameras`) + tilt 제약 RANSAC으로 벽/화분옆면 기각.
  - **정직한 한계**: `up_from_cameras`조차 "카메라를 똑바로 들었다"는 가정. 파이프라인에 IMU/중력센서가 없어 **어디에도 진짜 중력 GT는 없다.** 카메라-up은 더 나은 휴리스틱일 뿐.

## 4. PCA 재적합의 "각도 오차" — 무엇이고 믿을 수 있나
- **무엇**: 추정 법선 vs **합성 데이터에 내가 정해둔 진짜 법선** 사이 각도. 계산 `arccos(|n_추정·n_정답|)`.
- **신뢰도**:
  - ✅ **알고리즘 정밀도**로는 신뢰(평면 위 점에서 그 법선을 정확히 복원하는가). 0.018°.
  - ⚠️ **현실 정확도로는 아님** — 합성 GT 대비일 뿐, **실데이터엔 비교할 up GT가 없다**(3번). "우리 up이 0.018°로 정확"이 아니라 "PCA가 알려진 평면을 잘 푼다"는 의미.

## 5. RANSAC 초기평면 + 반복 PCA(TLS) 재적합
- RANSAC: 랜덤 3점으로 후보 평면을 수천 번 만들고 **인라이어 최다** 평면 채택(개별 3점이 바닥인지 모름 — 합의로 결정). 새 트랙은 "최저 z" 대신 **"법선이 up_prior 근처"** tilt 게이트로 후보 제한.
- 반복 PCA: RANSAC 3점 평면은 분산 큼 → 인라이어 전체에 PCA(공분산 최소 고유벡터=법선) **2~3회 반복** → 수렴(map-LAIcrop/ground.py에서 ~2°→0.05° 확인).

## 6. 해상도 vs LAI
- 728은 518보다 표면 디테일·점밀도 약간↑(render_depthmap 확인)이나, 모델은 **캐노피 표면만** 봄(내부 잎 못 뚫음). → **해상도는 LAI 정확도에 2차 레버**. 본질 한계는 표면-only.

## 7. 부피×LAD 접근 — 문헌 확인 + 참고답변 비판
- **문헌**: "부피 × 밀도계수"는 **Leaf Area Density(LAD, m²/m³)**라는 확립된 개념. 3D 점군→LAI에 표준적으로 쓰임(vineyard UAV, soybean 3D재건, horticulture lidar 등). 참고: 가로수 잎밀도 1.3–1.6 m²/m³.
- **참고답변(외부 LLM) 평가**:
  - ✅ hollow-shell, saturation(군락폐쇄), scale-drift, **allometric upscaling**(부피→파괴측정 LAI 회귀; 제일 견고).
  - ❌ **3방법 혼동**: `LAI=-ln(I/I₀)/k`(gap-fraction)와 voxel-LAD는 **투과/광선 데이터**(광센서·라이다 beam) 필요 → **passive RGB 재건엔 부적용**. 쓸 수 있는 건 **외피 부피 × LAD(∫LAD dz)**뿐(LAD는 k가 아니라 보정으로).
  - ⚠️ **스케일**: footprint 비율법은 스케일 상쇄, **부피법은 스케일 민감** → 화분 지름을 앵커로.
  - ⚠️ 인용 일부(coolenjoy/threads/facebook)는 출시 뉴스·소셜글 → 과학 근거 아님.

## 8. 최종 방향 결정
- **하이브리드**: LAI를 두 방식으로 산출·비교 — **footprint(silhouette, 스케일 강건 프록시)** vs **부피×LAD(숨은 biomass 추론, 스케일·보정 의존)**.
- **per-pot**, **새 트랙 `map-LAIpot`**(map-LAIcrop과 분리).
- up 벡터(Stage 1)는 두 트랙 공통 인프라.

---

## 열린 결정 (미정, 설정값으로 처리)
- **LAD 출처**: 작물별 문헌 기본값 vs allometric 보정(파괴측정 표본 필요 — 식재밀도와 함께 추후 결정).
- **스케일 앵커**: 화분 지름 파라미터 vs lingbot metric 스케일 신뢰.

## 신뢰 가능한 참고 문헌
- Vineyard LAI from UAV 3D point clouds — Springer (s11119-019-09699-x)
- Soybean canopy LAI from 3D reconstruction — ScienceDirect (S1574954123000997)
- Canopy Density in Horticulture via 3D Lidar SLAM — arXiv 2007.15652
- Mapping forest leaf area density from terrestrial lidar — Wiley (2041-210X.13550)
- Leaf area density 정의/단위 — envi-met KB:lad
