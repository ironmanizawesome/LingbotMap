"""map-LAIpot: 온실 화분작물 per-pot LAI 추정 트랙.

map-LAIcrop(노지/nadir)과 별도. 가로 촬영 + 베드 위 화분 가정.
면적은 평면 높이가 아니라 up 벡터(normal)에만 의존하므로, "지면 평면" 대신
robust up 벡터를 추정하고 화분별로 분리해 per-pot LAI를 낸다. OVERVIEW.md 참고.
"""

from .up_vector import estimate_up_vector, up_from_cameras

__all__ = ["estimate_up_vector", "up_from_cameras"]
