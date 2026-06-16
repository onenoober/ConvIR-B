# v2.11 Invalidated FSNet+UDP Preliminary Run

The first FSNet+UDP v2.11 attempt used raw DepthAnything V2 `.npy` depth cache
values directly and padded FSNet+UDP inference to factor `32`. This did not
match the official UDPNet test/data contract, which reads `test/depth2l/*.png`
through PIL `L` mode and uses factor-`8` padding in the official test and
validation scripts.

Those preliminary FSNet+UDP CSVs/log rows were deleted and must not be cited.
The final v2.11 FSNet+UDP evidence uses official-style `depth2l` PNGs generated
from the same DepthAnything V2 raw cache by per-image min-max normalization,
then read through the official `L` image path semantics, with factor-`8`
padding and strict checkpoint load after the documented `num_heads=1 -> 2`
builder patch.

The repaired endpoint reproduces the UDPNet README Haze4K reference within
rounding tolerance: final FSNet+UDP endpoint is `35.274720` PSNR and `0.990780`
endpoint SSIM versus the README table reference `35.31 / 0.99`.
