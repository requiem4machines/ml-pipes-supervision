from __future__ import annotations

from typing import Any, Generic, TypeVar, cast

import numpy as np
import numpy.typing as npt
import supervision as sv

from ml_pipes.operator import Operator
from ml_pipes.standard import SideEffectOp
from ml_pipes.vision import ImagePayload

PayloadT = TypeVar("PayloadT")


@Operator
class PlotImage(SideEffectOp[PayloadT]):
    def __init__(self, at: int | None = None) -> None:
        self.at = at

    def effect(self, payload: PayloadT) -> None:
        image: Any = payload[self.at] if self.at is not None else payload
        sv.plot_image(image)


@Operator
class FPSMonitor(SideEffectOp[PayloadT], Generic[PayloadT]):
    def __init__(self, sample_size: int = 30) -> None:
        self.monitor = sv.FPSMonitor(sample_size=sample_size)

    def reset(self) -> None:
        self.monitor.reset()

    def effect(self, payload: PayloadT) -> None:
        self.monitor.tick()
        print(f"\rFPS: {self.monitor.fps:.2f}", end="", flush=True)


@Operator
class ImageWindow(SideEffectOp[PayloadT], Generic[PayloadT]):
    def __init__(
        self,
        title: str = "supervision",
        keep_aspect_ratio: bool = True,
        at: int | None = None,
    ) -> None:
        self.title = title
        self.keep_aspect_ratio = keep_aspect_ratio
        self.at = at
        self.window = sv.ImageWindow(
            title=title,
            keep_aspect_ratio=keep_aspect_ratio,
        )

    def effect(self, payload: PayloadT) -> None:
        image: Any = payload[self.at] if self.at is not None else payload
        self.window.show(self._frame(image))

    @staticmethod
    def _payload_to_bgr_hwc(image: ImagePayload) -> npt.NDArray[np.uint8]:
        if image.layout != "HWC":
            raise ValueError(f"ImageWindow expects HWC images, got {image.layout!r}")

        converted = np.ascontiguousarray(image.array)
        if image.color_space == "RGB":
            return cast(npt.NDArray[np.uint8], converted[..., ::-1].copy())
        return cast(npt.NDArray[np.uint8], converted)

    @classmethod
    def _frame(cls, image: Any) -> npt.NDArray[np.uint8]:
        if isinstance(image, ImagePayload):
            return cls._payload_to_bgr_hwc(image)
        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"ImageWindow expects an ImagePayload or numpy.ndarray, got {type(image).__name__}"
            )
        return cast(npt.NDArray[np.uint8], np.ascontiguousarray(image))
