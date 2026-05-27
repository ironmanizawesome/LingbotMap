# Colab Drift Test — `travel` 시퀀스 재현

> **목적**: 지난번 관찰된 drift 현상 재현 + 정량/정성 기록 → PPT "LingBot-Map 한계" 슬라이드 근거
> **환경**: Colab T4 (무료), runtime → GPU 활성화
> **데이터**: `travel` 시퀀스 (603 frames, 518×294)
> **예상 시간**: 약 10분 추론 + 다운로드/시각화

---

## 셀 1 — Colab 환경 확인

```python
!nvidia-smi
import sys; print(sys.version)
```
> T4 / 15GB 메모리 / Python 3.10+ 확인.

---

## 셀 2 — Repo clone (demo.py 패치 포함됨, sed 필요 없음)

```python
%cd /content
!git clone https://github.com/ironmanizawesome/lingbot-map.git
%cd /content/lingbot-map
!git log --oneline -5
```
> `949841f feat(demo): add headless npz export and standalone viewer` 가 보이면 패치 포함된 것.
> **fork 주소 확인 필요** — 본인 GitHub 계정 fork URL 맞는지.

---

## 셀 3 — 의존성 설치

```python
!pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
!pip install -e .
!pip install -e ".[vis]"
!pip install onnxruntime-gpu
```
> PyTorch 2.8.0 + CUDA 12.8. FlashInfer는 안 깖 (SDPA fallback 쓸 거임 — T4에서 안정적).

---

## 셀 4 — 모델 체크포인트 다운로드

```python
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download(
    repo_id="robbyant/lingbot-map",
    filename="lingbot-map-long.pt",
    local_dir="/content/checkpoints",
)
print(f"Checkpoint: {ckpt}")
```
> ~4.6GB, 1~2분.

---

## 셀 5 — `travel` 데이터셋 다운로드

```python
from huggingface_hub import snapshot_download
data_dir = snapshot_download(
    repo_id="robbyant/lingbot-map-demo",
    repo_type="dataset",
    local_dir="/content/lingbot-map-demo",
)
print(f"Data: {data_dir}")
!ls /content/lingbot-map-demo/travel | head -5
!ls /content/lingbot-map-demo/travel | wc -l
```
> 11.4GB, 5~10분. `travel/` 폴더에 603장, `travel_sky_masks/`도 함께.

---

## 셀 6 — 추론 (지난번과 동일 설정, 재현성 확인)

```python
!python demo.py \
    --image_folder /content/lingbot-map-demo/travel \
    --model_path /content/checkpoints/lingbot-map-long.pt \
    --use_sdpa \
    --offload_to_cpu \
    --camera_num_iterations 4 \
    --keyframe_interval 2 \
    --mask_sky \
    --sky_mask_dir /content/lingbot-map-demo/travel_sky_masks \
    --save_predictions /content/travel_pred.npz
```

**관찰 포인트** (콘솔에서 메모해두기):
- 총 소요 시간 (지난번: 약 9분 55초)
- GPU peak 메모리 (지난번: 13.67/15GB)
- 어느 프레임쯤에서 처리가 눈에 띄게 느려지는지 (있다면)

---

## 셀 7 — 결과 npz 검증 (drift 정량 지표 미리 보기)

```python
import numpy as np
data = np.load("/content/travel_pred.npz", allow_pickle=True)
print("Keys:", list(data.files))

# 카메라 trajectory 추출
extrinsic = data["extrinsic"]   # (S, 4, 4) world-to-cam
print(f"extrinsic shape: {extrinsic.shape}")

# cam-to-world로 뒤집어서 카메라 위치만 추출
cam_positions = []
for E in extrinsic:
    R, t = E[:3, :3], E[:3, 3]
    cam_pos = -R.T @ t
    cam_positions.append(cam_pos)
cam_positions = np.array(cam_positions)  # (S, 3)

# 기본 통계
print(f"\n카메라 위치 범위:")
print(f"  X: [{cam_positions[:,0].min():.3f}, {cam_positions[:,0].max():.3f}]")
print(f"  Y: [{cam_positions[:,1].min():.3f}, {cam_positions[:,1].max():.3f}]")
print(f"  Z: [{cam_positions[:,2].min():.3f}, {cam_positions[:,2].max():.3f}]")

# 인접 프레임 간 이동 거리 (drift 의심 신호 1: 갑자기 큰 jump)
diffs = np.linalg.norm(np.diff(cam_positions, axis=0), axis=1)
print(f"\n프레임 간 이동 거리:")
print(f"  평균: {diffs.mean():.4f}m,  표준편차: {diffs.std():.4f}m")
print(f"  최대: {diffs.max():.4f}m  (frame {diffs.argmax()})")
print(f"  최소: {diffs.min():.4f}m")

# 시작점-끝점 거리 vs 누적 이동 거리 (drift 의심 신호 2)
total_path = diffs.sum()
start_end = np.linalg.norm(cam_positions[-1] - cam_positions[0])
print(f"\n누적 경로 길이: {total_path:.3f}m")
print(f"시작-끝 직선 거리: {start_end:.3f}m")
print(f"비율 (start-end / total): {start_end/total_path:.3f}")
```

> **해석**:
> - 인접 프레임 jump가 평균 대비 10배 이상 튀면 → pose 불연속 (drift 시작 후보)
> - 만약 `travel`이 loop 형태(끝점이 시작점으로 돌아오는)면 start-end 거리가 0에 가까워야 함
> - travel은 직선 이동 위주라 비율은 1에 가까워야 정상

---

## 셀 8 — Trajectory 시각화 (matplotlib, Colab 안에서 바로)

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(15, 5))

# 3D trajectory
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot(cam_positions[:,0], cam_positions[:,1], cam_positions[:,2], 'b-', alpha=0.5)
ax1.scatter(cam_positions[0,0], cam_positions[0,1], cam_positions[0,2], c='g', s=100, label='start')
ax1.scatter(cam_positions[-1,0], cam_positions[-1,1], cam_positions[-1,2], c='r', s=100, label='end')
ax1.set_title('Camera Trajectory (3D)')
ax1.legend()

# Top-down (X-Z plane)
ax2 = fig.add_subplot(132)
ax2.plot(cam_positions[:,0], cam_positions[:,2], 'b-', alpha=0.5)
ax2.scatter(cam_positions[0,0], cam_positions[0,2], c='g', s=100, label='start')
ax2.scatter(cam_positions[-1,0], cam_positions[-1,2], c='r', s=100, label='end')
ax2.set_title('Top-down (X-Z)')
ax2.set_aspect('equal')
ax2.legend()

# Inter-frame distances
ax3 = fig.add_subplot(133)
ax3.plot(diffs)
ax3.axhline(diffs.mean(), color='r', linestyle='--', alpha=0.5, label=f'mean={diffs.mean():.4f}')
ax3.set_xlabel('frame index')
ax3.set_ylabel('inter-frame distance (m)')
ax3.set_title('Frame-to-frame movement')
ax3.legend()

plt.tight_layout()
plt.savefig('/content/trajectory.png', dpi=120)
plt.show()
```

> **눈으로 확인할 것**:
> - 3D trajectory가 부드러운 곡선/직선인지, 중간에 꺾이거나 튀는지
> - 인접 프레임 거리 그래프에 spike가 있는지
> - **drift 있으면**: trajectory 후반부가 갑자기 휘거나, 같은 영역을 두 번 지나는데 안 겹침

---

## 셀 9 — 다운로드 (로컬 viser 시각화용)

```python
from google.colab import files
files.download('/content/travel_pred.npz')        # 메인 결과
files.download('/content/trajectory.png')         # trajectory 그림
```

> npz는 ~500MB~1GB 정도. 느리면 Drive에 옮겨놓고 받기.

---

## 로컬에서 — Viser로 점 분포 확인

```powershell
# Windows / conda env: lingbot-map
cd c:\Users\ironm\dev\lingbot-map
python view_npz.py C:\Users\ironm\Downloads\travel_pred.npz `
    --downsample_factor 2 `
    --point_size 0.0005 `
    --conf_threshold 1.0
```
> 브라우저 `http://localhost:8080`
> **drift 정성 관찰**:
> - 같은 영역(예: 도로/벽)이 두 겹으로 보이면 drift 확정
> - 카메라 trajectory(viser는 카메라 frustum도 표시함)가 후반부에 휘는지
> - 시작 부분 점들과 끝 부분 점들 색깔 비교 (이미지 RGB가 살아있음)

---

## 기록 양식 (관찰 후 메모리/PPT 반영용)

작업 끝나면 다음 항목을 채워서 알려주세요:

```
[기본]
- 추론 시간:
- GPU peak:
- 콘솔에 뜬 워닝/에러:

[Trajectory 분석 — 셀 7 출력]
- 인접 프레임 평균 거리:
- 최대 jump 발생 frame:
- 시작-끝 직선/누적 비율:

[Drift 정성 관찰 — 셀 8 그래프 + 로컬 viser]
- 3D trajectory 모양 (부드러운가? 꺾이는가? 후반부 휘는가?):
- 점 분포에 ghost/two-layer 보이는가? (frame 몇 번대부터?):
- 그 외 이상 (스케일 점프, 갑작스러운 회전 등):
```

이 기록 받으면:
1. `colab_workflow.md` 메모리 업데이트 (관찰 사실 추가)
2. PPT 원고에 "LingBot-Map 한계" 슬라이드 1장 추가 (실제 관찰 결과 + 가능한 원인 분석)
